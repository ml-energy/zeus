from zeus.device.cpu.rapl import RAPLFile, RAPL_DIR, RaplWraparoundTracker
import time
from unittest import mock

tmp_file = "/tmp/test_energy_uj"
def write(num):
    with open(tmp_file, 'w') as file:
        file.write(str(num))


builtin_open = open  # save the unpatched version

energy_uj_mock = 2000000
def mock_open(*args, **kwargs):
    if args[0] == RAPL_DIR+"/intel-rapl:0/energy_uj":
        # mocked open for path "foo"
        new_args = (tmp_file,)
        return open(*new_args, **kwargs)
    if args[0] == RAPL_DIR+"/intel-rapl:0/max_energy_range_uj":
        return mock.mock_open(read_data="10000000")(*args, **kwargs)
    # unpatched version for every other path
    return builtin_open(*args, **kwargs)

@mock.patch("builtins.open", mock_open)
def main():
    rapl_monitor = RaplWraparoundTracker(
        tmp_file,
        10000000
    )
    write(2000000)
    time.sleep(2)
    write(1000000)
    time.sleep(1)
    print("Num wraparounds", rapl_monitor.get_num_wraparounds())
    assert(rapl_monitor.get_num_wraparounds() == 1)

if __name__ == "__main__":
    main()
