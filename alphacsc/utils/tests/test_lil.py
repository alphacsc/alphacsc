import numpy as np

from numpy.testing import assert_allclose

from alphacsc.utils.lil import convert_to_list_of_lil
from alphacsc.utils.lil import convert_from_list_of_lil
from alphacsc.utils.lil import safe_sum, get_z_shape, scale_z_by_atom
from alphacsc.utils.lil import is_list_of_lil, is_lil


def test_is_list_of_lil():
    n_trials, n_atoms, n_times_valid = 3, 2, 10
    z = np.random.randn(n_trials, n_atoms, n_times_valid)
    z_lil = convert_to_list_of_lil(z)

    assert is_list_of_lil(z_lil)
    assert not is_list_of_lil(z)
    assert is_lil(z_lil[0])
    assert not is_lil(z[0])


def test_get_z_shape():
    n_trials, n_atoms, n_times_valid = 3, 2, 10
    z = np.random.randn(n_trials, n_atoms, n_times_valid)
    z_lil = convert_to_list_of_lil(z)
    assert_allclose(get_z_shape(z), get_z_shape(z_lil))


def test_safe_sum():
    n_trials, n_atoms, n_times_valid = 3, 2, 10
    z = np.random.randn(n_trials, n_atoms, n_times_valid)
    z_lil = convert_to_list_of_lil(z)
    for axis in [None, (0, 2)]:
        assert_allclose(safe_sum(z, axis), safe_sum(z_lil, axis))


def test_conversion():
    n_trials, n_atoms, n_times_valid = 3, 2, 10
    z = np.random.randn(n_trials, n_atoms, n_times_valid)
    z_lil = convert_to_list_of_lil(z)
    z_2 = convert_from_list_of_lil(z_lil)
    assert_allclose(z, z_2)


def test_scale_z_by_atom():
    n_trials, n_atoms, n_times_valid = 3, 2, 10
    scale = np.random.randn(n_atoms)
    z = np.random.randn(n_trials, n_atoms, n_times_valid)
    z_lil = convert_to_list_of_lil(z)
    z_scaled = scale_z_by_atom(z, scale)
    z_lil_scaled = scale_z_by_atom(z_lil, scale)
    assert_allclose(z_scaled, convert_from_list_of_lil(z_lil_scaled))
