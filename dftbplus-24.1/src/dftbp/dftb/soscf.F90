!--------------------------------------------------------------------------------------------------!
!  DFTB+: general package for performing fast atomistic simulations                                !
!  Copyright (C) 2006 - 2023  DFTB+ developers group                                               !
!                                                                                                  !
!  See the LICENSE file for terms of usage and distribution.                                       !
!--------------------------------------------------------------------------------------------------!

!> Second-Order SCF (SOSCF) orbital rotation optimiser — L-SR1 quasi-Newton.
!!
!! Algorithm:
!!   Pre-SOSCF: level-shifted Delta-SCF until max|g| < threshold.
!!   SOSCF:     direct orbital rotation  C_{n+1} = C_n * exp(kappa)
!!              with quasi-Newton step   Delta_x = -H * g
!!              and L-SR1 inverse-Hessian update.
!!
!! The L-SR1 inverse Hessian is one matrix stacked over both spin channels,
!! so the secant pair (delta, gamma) carries cross-spin coupling.
module dftbp_dftb_soscf
  use dftbp_common_accuracy, only : dp
  use dftbp_math_blasroutines, only : gemm, gemv, ger
  use dftbp_math_lapackroutines, only : gesv
  implicit none
  private

  public :: TSoscfSpin, TSoscf
  public :: soscf_init, computeOrbGradient, getSoscfStep
  public :: buildKappa, computeCayleyExp, updateInvHessianLSR1
  public :: getInvHessDiag


  !> Relative tolerance for skipping the L-SR1 update (safeguard against
  !! near-zero denominators that would cause numerical blow-up).
  real(dp), parameter :: lsr1SkipTol = 1.0e-8_dp

  !> Minimum step-size norm for the L-SR1 update.
  !! If the previous step delta = DxPrev is smaller than this (the gradient was
  !! already near-zero before the rotation), the secant pair (delta, gamma) carries
  !! no meaningful curvature information.  Applying the update would amplify numerical
  !! noise and blow up the inverse Hessian (rank-1 update magnitude ~ ||delta||^2 /
  !! |denom| diverges when both delta and gamma -> 0 together).
  real(dp), parameter :: lsr1MinStepNorm = 1.0e-5_dp

  ! Trust-region cap on the orbital rotation step is configurable at runtime
  ! via TSoscf%useMaxKappa and TSoscf%maxKappa (see below).  Motivation:
  !
  !   The diagonal initial inverse Hessian H^{-1}_{ia,ia} = sign(gap)/max(2|gap|,1e-3)
  !   ignores off-diagonal Coulomb/exchange couplings between occ-virt pairs.  For
  !   high-order excited-state saddles (multiple negative-curvature modes, as in
  !   double excitations or non-adjacent multi-orbital excitations) the resulting
  !   unconstrained Newton step Dx = -H^{-1} g can exceed 1 rad, invalidating the
  !   Cayley linearisation, scrambling IMOM assignments, and poisoning subsequent
  !   L-SR1 updates with garbage secant pairs.
  !
  !   Capping max|Dx| at a physically reasonable rotation angle preserves the step
  !   direction but prevents overshoot.  For well-behaved cases (HOMO-LUMO, moderate
  !   multi-saddle) the cap is inactive and convergence is unaffected.


  !> State for one spin channel.
  type :: TSoscfSpin

    !> Number of occupied / virtual orbitals for this spin
    integer :: nOcc = 0
    integer :: nVirt = 0
    integer :: nOccVirt = 0

    !> Offset of this spin's slice in the parent TSoscf stacked vectors
    !! (this spin owns indices offset+1 .. offset+nOccVirt).
    integer :: offset = 0

    !> 1-based indices of occupied orbitals in the full orbital list
    integer, allocatable :: occIdx(:)

    !> 1-based indices of virtual orbitals in the full orbital list
    integer, allocatable :: virtIdx(:)

    !> Accumulated orbital rotation vector x, length nOcc*nVirt.
    !! Ordering: x( (iOcc-1)*nVirt + iVirt ) = rotation angle for pair (iOcc, iVirt).
    real(dp), allocatable :: xVec(:)

    !> True once soscf_init has been called for this spin.
    logical :: isInitialized = .false.

  end type TSoscfSpin


  !> Top-level SOSCF state (holds one TSoscfSpin per spin channel).
  type :: TSoscf

    integer :: nSpin = 0

    !> Total length of the stacked occ-virt parameter vector,
    !! nTot = sum_sigma nOccVirt^sigma.
    integer :: nTot = 0

    !> Per-spin data, size nSpin.
    type(TSoscfSpin), allocatable :: spin(:)

    !> Orbital gradient from the previous SOSCF iteration, stacked over both
    !! spins, length nTot (for gamma = g_n - g_{n-1}).
    real(dp), allocatable :: gPrev(:)

    !> Quasi-Newton step from the previous SOSCF iteration, stacked over both
    !! spins, length nTot.  Used as the secant displacement delta in the
    !! L-SR1 update at the next outer iteration.
    real(dp), allocatable :: DxPrev(:)

    !> Approximate inverse Hessian, shape (nTot, nTot).
    !! Stored as a full symmetric matrix; both triangles are kept up to date.
    real(dp), allocatable :: invHess(:,:)

    !> Gradient threshold for entering SOSCF (and for final convergence).
    real(dp) :: threshold = 1.0e-4_dp

    !> Trust-region clip: if max|Dx| > maxKappa after the Newton step, Dx is
    !! rescaled so max|Dx| = maxKappa, preserving the step direction.
    logical  :: useMaxKappa = .true.

    !> Cap magnitude in radians (default 0.5 rad ~ 29 deg).
    real(dp) :: maxKappa = 0.5_dp

  end type TSoscf


contains


  !> Initialise the SOSCF state for all spin channels.
  !!
  !! Determines the occ/virt partition from the non-aufbau target filling,
  !! allocates all per-spin arrays, and sets the initial diagonal inverse
  !! Hessian H_ia,ia = 1 / (epsilon_a - epsilon_i).
  subroutine soscf_init(this, nSpin, nOrb, eigen, targetFilling, threshold, &
      & useMaxKappa, maxKappa)

    !> SOSCF state to initialise (intent(out) clears any previous state).
    type(TSoscf), intent(out) :: this

    !> Number of spin channels (1 or 2).
    integer, intent(in) :: nSpin

    !> Number of orbitals.
    integer, intent(in) :: nOrb

    !> Orbital eigenvalues (nOrb, 1, nSpin).
    real(dp), intent(in) :: eigen(:,:,:)

    !> Non-aufbau target filling that defines the excited-state occ/virt
    !! partition, shape (nOrb, 1, nSpin).  Occupation > 0.5 = occupied.
    real(dp), intent(in) :: targetFilling(:,:,:)

    !> Gradient threshold for entering / exiting SOSCF.
    real(dp), intent(in) :: threshold

    !> Enable trust-region clip on the orbital rotation step.
    logical, intent(in) :: useMaxKappa

    !> Cap magnitude in radians (ignored if useMaxKappa = .false.).
    real(dp), intent(in) :: maxKappa

    integer :: iSpin, iOrb, iOcc, iVirt, k, kAbs, runOffset
    integer :: nOcc, nVirt

    this%nSpin       = nSpin
    this%threshold   = threshold
    this%useMaxKappa = useMaxKappa
    this%maxKappa    = maxKappa
    allocate(this%spin(nSpin))

    ! --- count occ/virt per spin and assign offsets into the stacked vectors -
    runOffset = 0
    do iSpin = 1, nSpin
      nOcc  = count(targetFilling(:,1,iSpin) > 0.5_dp)
      nVirt = nOrb - nOcc
      this%spin(iSpin)%nOcc     = nOcc
      this%spin(iSpin)%nVirt    = nVirt
      this%spin(iSpin)%nOccVirt = nOcc * nVirt
      this%spin(iSpin)%offset   = runOffset
      runOffset = runOffset + nOcc * nVirt
    end do
    this%nTot = runOffset

    ! --- allocate stacked secant buffers and inverse Hessian -----------------
    allocate(this%gPrev (this%nTot))
    allocate(this%DxPrev(this%nTot))
    allocate(this%invHess(this%nTot, this%nTot))
    this%gPrev (:)    = 0.0_dp
    this%DxPrev(:)    = 0.0_dp
    this%invHess(:,:) = 0.0_dp

    do iSpin = 1, nSpin

      nOcc  = this%spin(iSpin)%nOcc
      nVirt = this%spin(iSpin)%nVirt

      ! --- allocate per-spin index lists and rotation vector ----------------
      allocate(this%spin(iSpin)%occIdx(nOcc))
      allocate(this%spin(iSpin)%virtIdx(nVirt))
      allocate(this%spin(iSpin)%xVec(nOcc * nVirt))

      ! --- collect occ/virt orbital indices ---------------------------------
      iOcc = 0
      iVirt = 0
      do iOrb = 1, nOrb
        if (targetFilling(iOrb, 1, iSpin) > 0.5_dp) then
          iOcc = iOcc + 1
          this%spin(iSpin)%occIdx(iOcc) = iOrb
        else
          iVirt = iVirt + 1
          this%spin(iSpin)%virtIdx(iVirt) = iOrb
        end if
      end do

      ! --- initialise rotation vector ---------------------------------------
      this%spin(iSpin)%xVec(:) = 0.0_dp

      ! --- initialise diagonal inverse Hessian: H_ia,ia = 1/(2*(e_a - e_i)) ---
      !
      ! From the ORCA formalism, the diagonal orbital Hessian is:
      !   d^2E/d(kappa_ia)^2 = 2*(f_a - f_i)*(e_i - e_a)
      ! For integer occupations (f_occ=1, f_virt=0): = 2*(e_a - e_i) = 2*gap
      ! So the inverse Hessian diagonal is 1/(2*gap).
      !
      ! The sign of gap matters for excited-state saddle-point convergence:
      !   gap > 0  (e_a > e_i): excited state is a LOCAL MINIMUM in this direction.
      !            Standard descent step  Dx = -invH * g  converges correctly.
      !   gap < 0  (e_a < e_i): excited state is a SADDLE POINT in this direction
      !            (occurs for non-HOMO-LUMO excitations where the hole orbital
      !            sits below a still-occupied orbital in energy).
      !            A Newton step with the SIGNED inverse Hessian (invH < 0)
      !            automatically points toward the saddle-point stationary point.
      !            Using abs() would flip the sign and drive the system toward
      !            the ground-state minimum instead (variational collapse).
      !
      ! Therefore: use the SIGNED gap with a magnitude clamp to avoid division
      ! by zero at near-degenerate pairs.
      k = 0
      do iOcc = 1, nOcc
        do iVirt = 1, nVirt
          k = k + 1
          kAbs = this%spin(iSpin)%offset + k
          associate( gap => eigen(this%spin(iSpin)%virtIdx(iVirt), 1, iSpin) &
              &             - eigen(this%spin(iSpin)%occIdx(iOcc),  1, iSpin) )
            this%invHess(kAbs, kAbs) = &
                & sign(1.0_dp, gap) / max(2.0_dp * abs(gap), 1.0e-3_dp)
          end associate
        end do
      end do

      this%spin(iSpin)%isInitialized = .true.

    end do

  end subroutine soscf_init


  !> Compute the orbital gradient for one spin channel.
  !!
  !! F_MO = C^T * H_AO * C  (transform Fock to MO basis)
  !! g_ia = 4 * F_MO( occIdx(i), virtIdx(a) )
  !!
  !! H_AO must be the full symmetric physical Hamiltonian (both triangles
  !! populated — call symmetrizeHS before this routine if needed).
  subroutine computeOrbGradient(eigvecs, Hao, nOrb, occIdx, virtIdx, &
      & nOcc, nVirt, g)

    !> Eigenvectors, shape (nOrb, nOrb): column i = i-th MO in AO basis.
    real(dp), intent(in) :: eigvecs(:,:)

    !> Dense AO Hamiltonian (nOrb, nOrb), full symmetric.
    real(dp), intent(in) :: Hao(:,:)

    !> Total number of orbitals.
    integer, intent(in) :: nOrb

    !> Indices of occupied orbitals (1-based), length nOcc.
    integer, intent(in) :: occIdx(:)

    !> Indices of virtual orbitals (1-based), length nVirt.
    integer, intent(in) :: virtIdx(:)

    integer, intent(in) :: nOcc, nVirt

    !> Orbital gradient, length nOcc*nVirt.
    !! Ordering: g( (iOcc-1)*nVirt + iVirt ).
    real(dp), intent(out) :: g(:)

    real(dp), allocatable :: HC(:,:), Fmo(:,:)
    integer :: iOcc, iVirt, k

    allocate(HC(nOrb, nOrb))
    allocate(Fmo(nOrb, nOrb))

    ! HC = H_AO * C
    call gemm(HC, Hao, eigvecs)

    ! F_MO = C^T * HC
    call gemm(Fmo, eigvecs, HC, transA='T')

    ! Extract occ-virt block and scale by 4
    k = 0
    do iOcc = 1, nOcc
      do iVirt = 1, nVirt
        k = k + 1
        g(k) = 4.0_dp * Fmo(occIdx(iOcc), virtIdx(iVirt))
      end do
    end do

    deallocate(HC, Fmo)

  end subroutine computeOrbGradient


  !> Compute the quasi-Newton step Delta_x = -H * g over the stacked
  !! occ-virt vector (length nTot).
  !!
  !! Followed by an optional trust-region clip: if soscf%useMaxKappa and
  !! max|Dx| > soscf%maxKappa, Dx is rescaled so max|Dx| = soscf%maxKappa,
  !! preserving the step direction.
  subroutine getSoscfStep(soscf, g, Dx)

    !> Top-level SOSCF state (supplies invHess and useMaxKappa / maxKappa).
    type(TSoscf), intent(in) :: soscf

    !> Current orbital gradient, length nTot.
    real(dp), intent(in) :: g(:)

    !> Quasi-Newton step, length nTot.
    real(dp), intent(out) :: Dx(:)

    real(dp) :: maxAbsDx

    ! Dx = -invHess * g
    call gemv(Dx, soscf%invHess, g, alpha=-1.0_dp, beta=0.0_dp)

    ! Trust-region clip (optional).
    if (soscf%useMaxKappa) then
      maxAbsDx = maxval(abs(Dx))
      if (maxAbsDx > soscf%maxKappa) then
        Dx(:) = Dx(:) * (soscf%maxKappa / maxAbsDx)
      end if
    end if

  end subroutine getSoscfStep


  !> Build the antisymmetric orbital rotation matrix kappa in MO space.
  !!
  !! kappa( occIdx(i), virtIdx(a) ) =  x( (i-1)*nVirt + a )
  !! kappa( virtIdx(a), occIdx(i) ) = -x( (i-1)*nVirt + a )
  !! All other elements are zero.
  subroutine buildKappa(x, occIdx, virtIdx, nOcc, nVirt, nOrb, kappa)

    !> Accumulated rotation vector, length nOcc*nVirt.
    real(dp), intent(in) :: x(:)

    !> Occupied orbital indices (1-based), length nOcc.
    integer, intent(in) :: occIdx(:)

    !> Virtual orbital indices (1-based), length nVirt.
    integer, intent(in) :: virtIdx(:)

    integer, intent(in) :: nOcc, nVirt, nOrb

    !> Antisymmetric rotation matrix, shape (nOrb, nOrb).
    real(dp), intent(out) :: kappa(:,:)

    integer :: iOcc, iVirt, k

    kappa(:,:) = 0.0_dp
    k = 0
    do iOcc = 1, nOcc
      do iVirt = 1, nVirt
        k = k + 1
        kappa(occIdx(iOcc), virtIdx(iVirt)) =  x(k)
        kappa(virtIdx(iVirt), occIdx(iOcc)) = -x(k)
      end do
    end do

  end subroutine buildKappa


  !> Compute the orbital rotation matrix U = exp(kappa) using the
  !! first-order Cayley (Padé [1,1]) approximant:
  !!
  !!   U = (I - kappa/2)^{-1} * (I + kappa/2)
  !!
  !! This is exact for any 2x2 antisymmetric block and accurate for small
  !! rotation angles.  It guarantees that U is orthogonal.
  !!
  !! Implementation: solve the linear system
  !!   (I - kappa/2) * U = (I + kappa/2)
  !! treating the nOrb columns of the RHS as nOrb separate right-hand sides.
  subroutine computeCayleyExp(kappa, nOrb, U)

    !> Antisymmetric rotation matrix, shape (nOrb, nOrb).
    real(dp), intent(in) :: kappa(:,:)

    !> Total number of orbitals.
    integer, intent(in) :: nOrb

    !> Orthogonal rotation matrix U = exp(kappa), shape (nOrb, nOrb).
    real(dp), intent(out) :: U(:,:)

    real(dp), allocatable :: Amtx(:,:)
    integer :: ii, iError

    allocate(Amtx(nOrb, nOrb))

    ! RHS = I + kappa/2  (stored in U so gesv overwrites it with the solution)
    U(:,:) = 0.5_dp * kappa
    do ii = 1, nOrb
      U(ii, ii) = U(ii, ii) + 1.0_dp
    end do

    ! LHS = I - kappa/2
    Amtx(:,:) = -0.5_dp * kappa
    do ii = 1, nOrb
      Amtx(ii, ii) = Amtx(ii, ii) + 1.0_dp
    end do

    ! Solve Amtx * U = RHS  (gesv overwrites U with the solution)
    call gesv(Amtx, U, iError=iError)
    if (iError /= 0) then
      ! Fallback: return identity if solver fails (should not happen)
      U(:,:) = 0.0_dp
      do ii = 1, nOrb
        U(ii, ii) = 1.0_dp
      end do
    end if

    deallocate(Amtx)

  end subroutine computeCayleyExp


  !> Update the inverse Hessian using the L-SR1 formula:
  !!
  !!   j   = delta - H * gamma
  !!   H  += j * j^T / (j^T * gamma)
  !!
  !! where delta = x_{n+1} - x_n = Delta_x
  !!       gamma = g_{n+1} - g_n
  !!
  !! The update is skipped when |j^T * gamma| is below the safety threshold
  !! lsr1SkipTol * ||j|| * ||gamma|| to avoid numerical blow-up.
  subroutine updateInvHessianLSR1(this, delta, gamma, deltaNorm, gammaNorm, denom_out, tUpdated)

    !> SOSCF state (invHess updated in-place).
    type(TSoscf), intent(inout) :: this

    !> Step taken: delta = Delta_x = x_{n+1} - x_n, length nTot.
    real(dp), intent(in) :: delta(:)

    !> Gradient change: gamma = g_{n+1} - g_n, length nTot.
    real(dp), intent(in) :: gamma(:)

    !> Diagnostic: ||delta||
    real(dp), intent(out) :: deltaNorm

    !> Diagnostic: ||gamma||
    real(dp), intent(out) :: gammaNorm

    !> Diagnostic: denominator j^T * gamma (0 if update was skipped)
    real(dp), intent(out) :: denom_out

    !> Diagnostic: .true. if the rank-1 update was actually applied
    logical, intent(out) :: tUpdated

    real(dp), allocatable :: jVec(:)
    real(dp) :: denom, jNorm, gNorm

    allocate(jVec(this%nTot))

    deltaNorm = sqrt(dot_product(delta, delta))
    gammaNorm = sqrt(dot_product(gamma, gamma))
    denom_out = 0.0_dp
    tUpdated  = .false.

    ! Skip if the previous step was negligibly small: the secant pair (delta, gamma)
    ! carries no curvature information when the gradient was already near-zero before
    ! the rotation.  Applying the rank-1 update in this regime amplifies floating-point
    ! noise and can cause ||invHess|| to blow up by factors of 10^6.
    if (deltaNorm < lsr1MinStepNorm) then
      deallocate(jVec)
      return
    end if

    ! j = delta - H * gamma
    call gemv(jVec, this%invHess, gamma, alpha=-1.0_dp, beta=0.0_dp)
    jVec(:) = delta(:) + jVec(:)    ! j = delta - H*gamma

    ! Denominator j^T * gamma
    denom = dot_product(jVec, gamma)
    denom_out = denom

    ! Safety check: skip if denominator is too small relative to vector norms
    jNorm = sqrt(dot_product(jVec, jVec))
    gNorm = sqrt(dot_product(gamma, gamma))
    if (abs(denom) < lsr1SkipTol * jNorm * gNorm + epsilon(1.0_dp)) then
      deallocate(jVec)
      return
    end if

    ! Rank-1 update: H += j * j^T / denom
    tUpdated = .true.
    call ger(this%invHess, 1.0_dp / denom, jVec, jVec)

    deallocate(jVec)

  end subroutine updateInvHessianLSR1


  !> Return the diagonal of the inverse Hessian for one spin channel.
  subroutine getInvHessDiag(this, iSpin, diagH)
    type(TSoscf), intent(in) :: this
    integer, intent(in) :: iSpin
    real(dp), intent(out) :: diagH(:)
    integer :: k, kAbs
    do k = 1, this%spin(iSpin)%nOccVirt
      kAbs = this%spin(iSpin)%offset + k
      diagH(k) = this%invHess(kAbs, kAbs)
    end do
  end subroutine getInvHessDiag


end module dftbp_dftb_soscf
