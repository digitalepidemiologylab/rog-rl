import pytest
import rog_rl.cli as cli


def test_demo(runner):
    result = runner.invoke(cli.demo, input="\n")
    assert result.exit_code == 0
    print(result.output)


def test_performance(runner):
    result = runner.invoke(cli.profile_perf, input="\n")
    assert result.exit_code == 0
    print(result.output)


@pytest.mark.skip(reason="Slows down testing")
def test_performance_render_on(runner):
    result = runner.invoke(cli.profile_perf, ["-ron"])
    assert result.exit_code == 0
    print(result.output)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
