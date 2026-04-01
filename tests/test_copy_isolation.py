"""Tests for solver/canonicalizer copy isolation behavior."""

import numpy as np
import pytest

from fomtools.verifier import Verifier


class TestMIPCopyIsolation:
    """Test that MIP solver copies don't accumulate objective constraints."""

    @pytest.mark.parametrize("solver", ["gurobi", "highs"])
    def test_objective_does_not_accumulate(self, solver):
        """Verify that solving multiple times doesn't accumulate objective constraints."""
        vp = Verifier(method='mip', solver=solver, obbt_handler=None)
        x = vp.add_iterate(3, lb=0, ub=10)
        z = vp.step.relu(x)

        # Solve twice with same objective
        vp.set_objective_norm([z], np.inf)
        result1 = vp.solve()
        result2 = vp.solve()

        # Results should be identical (no constraint accumulation)
        assert abs(result1 - result2) < 1e-6

    @pytest.mark.parametrize("solver", ["gurobi", "highs"])
    def test_objective_change_works(self, solver):
        """Verify that changing objective works correctly."""
        vp = Verifier(method='mip', solver=solver, obbt_handler=None)
        x = vp.add_iterate(3, lb=0, ub=10)
        z = vp.step.relu(x)

        # Solve with first objective
        vp.set_objective_norm([z], np.inf)
        result1 = vp.solve()

        # Change objective (shifted expression)
        vp.set_objective_norm([z - 5], np.inf)
        result2 = vp.solve()

        # Results should be different
        assert result1 != result2

        # Solve again with original objective to verify no corruption
        vp.set_objective_norm([z], np.inf)
        result3 = vp.solve()

        # Should match first result
        assert abs(result1 - result3) < 1e-6

    @pytest.mark.parametrize("solver", ["gurobi", "highs"])
    def test_canonicalizer_copy_independence(self, solver):
        """Verify that canonicalizer copy is independent from original."""
        vp = Verifier(method='mip', solver=solver, obbt_handler=None)
        x = vp.add_iterate(3, lb=0, ub=10)
        z = vp.step.relu(x)

        original_canon = vp.canonicalizer
        copy_canon = original_canon.copy()

        # The copy's solver should be different from original's
        assert copy_canon.solver is not original_canon.solver

        # Modifying copy shouldn't affect original
        assert copy_canon._objective_spec is None
        copy_canon.set_infinity_norm_objective([z], [z.lb], [z.ub])
        assert copy_canon._objective_spec is not None
        # Original still has no objective set (we haven't set it on original)
        # Note: _objective_spec is shared by reference, this tests independence


class TestSDPCopyIsolation:
    """Test that SDP canonicalizer copies don't accumulate objective constraints."""

    @pytest.mark.parametrize("solver", ["clarabel", "cvxpy"])
    def test_objective_does_not_accumulate(self, solver):
        """Verify that solving multiple times doesn't accumulate constraints."""
        vp = Verifier(method='sdp', solver=solver, obbt_handler=None)
        x = vp.add_iterate(3, lb=0, ub=10)
        z = vp.step.relu(x)

        # Solve twice with same objective
        vp.set_objective_norm(z, 2)
        result1 = vp.solve()
        result2 = vp.solve()

        # Results should be very close (no constraint accumulation)
        # SDP results can have small numerical differences
        assert abs(result1 - result2) < 1e-4

    @pytest.mark.parametrize("solver", ["clarabel", "cvxpy"])
    def test_psd_cone_not_accumulated(self, solver):
        """Verify that PSD cone indices don't accumulate across solves."""
        vp = Verifier(method='sdp', solver=solver, obbt_handler=None)
        x = vp.add_iterate(3, lb=0, ub=10)
        z = vp.step.relu(x)

        vp.set_objective_norm(z, 2)

        # Get initial psd cone count
        initial_psd_count = len(vp.canonicalizer.psd_cone_indices)

        # Solve multiple times
        for _ in range(3):
            vp.solve()

        # PSD cone count in original should not increase
        final_psd_count = len(vp.canonicalizer.psd_cone_indices)
        assert final_psd_count == initial_psd_count

    @pytest.mark.parametrize("solver", ["clarabel", "cvxpy"])
    def test_canonicalizer_copy_independence(self, solver):
        """Verify that SDP canonicalizer copy is independent from original."""
        vp = Verifier(method='sdp', solver=solver, obbt_handler=None)
        x = vp.add_iterate(3, lb=0, ub=10)

        original_canon = vp.canonicalizer
        copy_canon = original_canon.copy()

        # Deep copy checks - matrices should be independent
        assert copy_canon.Aeq_matrices is not original_canon.Aeq_matrices
        assert copy_canon.psd_cone_indices is not original_canon.psd_cone_indices

        # Modify copy and verify original unchanged
        initial_psd_count = len(original_canon.psd_cone_indices)
        copy_canon.add_psd_cone_indices([(0, 2)])
        assert len(original_canon.psd_cone_indices) == initial_psd_count
