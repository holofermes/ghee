"""Scene arguments: what a controller declares, how the tokens after `--` reach
build(), and what the bundled examples do with theirs. Pure Python except the
one render that checks an argument changes the frame (skips without a GL device).
"""

import argparse
import subprocess
import sys

import numpy as np
import pytest

from ghee import Arg, load_scene
from ghee.console import examples_dir
from ghee.console.store import SceneStore
from ghee.procedural import check_scene_args, declared_args

DECLARING_PY = """
from ghee import Arg, ArrayInput, Pass, Scene, SceneController

class Demo(SceneController):
    ARGS = [
        Arg("emit", "size", default=(8, 8), help="the pass size"),
        Arg("word", default="hi"),
        Arg("count", "int", default=3),
        Arg("dither", "bool", default=True),
    ]

    def build(self, base):
        w, h = self.args.emit
        return Scene(
            filters=["dither"] if self.args.dither else [],
            passes=[Pass(name="main", code=f"// {self.args.word} {self.args.count}\\n"
                                          "void main(){}",
                         arrays=[ArrayInput("vals", (w, h))])],
        )
"""

SILENT_PY = """
from ghee import SceneController

class Feed(SceneController):
    def provide(self, name, size, frame):
        return [1.0] * size
"""

JSON_SCENE = (
    '{"version":1,"passes":[{"name":"main","code":"void main(){}",'
    '"arrays":[{"name":"level","size":2}]}]}'
)


def declaring(tmp_path):
    (tmp_path / "demo.py").write_text(DECLARING_PY)
    return str(tmp_path / "demo.py")


def test_declared_defaults_apply_when_absent(tmp_path):
    scene, ctrl = load_scene(declaring(tmp_path))
    assert scene.passes[0].arrays[0].size == (8, 8)
    assert ctrl.args.word == "hi"
    assert ctrl.args.count == 3
    assert ctrl.args.dither is True
    assert scene.filters == ["dither"]


def test_args_reach_build(tmp_path):
    # the array size is frozen in build(), before any renderer exists
    scene, ctrl = load_scene(
        declaring(tmp_path), ["--emit", "16x9", "--word", "yo", "--count", "7"]
    )
    assert scene.passes[0].arrays[0].size == (16, 9)
    assert "// yo 7" in scene.passes[0].code
    assert ctrl.args.emit == (16, 9)


def test_bool_arg_takes_the_no_form(tmp_path):
    scene, ctrl = load_scene(declaring(tmp_path), ["--no-dither"])
    assert ctrl.args.dither is False
    assert scene.filters == []


def test_undeclared_arg_is_an_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        load_scene(declaring(tmp_path), ["--emmit", "16x9"])
    assert e.value.code == 2
    assert "unrecognized arguments: --emmit" in capsys.readouterr().err


def test_abbreviated_arg_is_an_error(tmp_path, capsys):
    with pytest.raises(SystemExit):
        load_scene(declaring(tmp_path), ["--em", "16x9"])
    assert "unrecognized arguments: --em" in capsys.readouterr().err


def test_bad_size_value_is_an_error(tmp_path, capsys):
    for bad in ("banana", "16", "0x9", "-4x9"):
        with pytest.raises(SystemExit):
            load_scene(declaring(tmp_path), ["--emit", bad])
        assert "WxH" in capsys.readouterr().err


def test_help_lists_the_declared_args(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        load_scene(declaring(tmp_path), ["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "ghee render demo --" in out  # usage names the scene
    assert "--emit WxH" in out
    assert "the pass size (default: 8x8)" in out
    assert "--dither, --no-dither" in out


def test_scene_declaring_nothing_still_loads(tmp_path):
    (tmp_path / "quiet.py").write_text(SILENT_PY)
    (tmp_path / "quiet.scene.json").write_text(JSON_SCENE)
    scene, ctrl = load_scene(str(tmp_path / "quiet.py"))
    assert ctrl.provide("level", 2, 0) == [1.0, 1.0]
    assert ctrl.args == argparse.Namespace()  # nothing declared, nothing parsed
    with pytest.raises(SystemExit):
        load_scene(str(tmp_path / "quiet.py"), ["--emit", "8x8"])


def test_json_scene_takes_no_args(tmp_path, capsys):
    (tmp_path / "flat.scene.json").write_text(JSON_SCENE)
    scene, ctrl = load_scene(str(tmp_path / "flat.scene.json"))
    assert ctrl is None
    with pytest.raises(SystemExit):
        load_scene(str(tmp_path / "flat.scene.json"), ["--emit", "8x8"])
    assert "unrecognized arguments" in capsys.readouterr().err


def test_lenient_load_keeps_what_the_scene_declares(tmp_path):
    # a host with one set of args and many scenes: this one takes --emit only
    scene, ctrl = load_scene(
        declaring(tmp_path), ["--emit", "4x4", "--text", "elsewhere"], strict=False
    )
    assert scene.passes[0].arrays[0].size == (4, 4)
    assert not hasattr(ctrl.args, "text")


def test_controller_built_by_hand_sees_the_defaults():
    from ghee import Pass, Scene, SceneController

    class Solo(SceneController):
        ARGS = [Arg("emit", "size", default=(4, 4))]

        def build(self, base):
            return Scene(passes=[Pass(name="main", code="void main(){}")])

    c = Solo()
    assert c.args.emit == (4, 4)
    c.args = argparse.Namespace(emit=(9, 9))  # a loader's values win
    assert c.args.emit == (9, 9)
    assert SceneController().args == argparse.Namespace()


def test_bad_declarations_are_rejected():
    with pytest.raises(ValueError, match="unknown type"):
        Arg("emit", "wxh")
    with pytest.raises(ValueError, match="has to be a word"):
        Arg("--emit")
    with pytest.raises(ValueError, match="has to be a word"):
        Arg("two words")
    with pytest.raises(ValueError, match="taken by the help option"):
        Arg("help")


def test_duplicate_declaration_reports_the_name(tmp_path):
    (tmp_path / "dup.py").write_text(
        "from ghee import Arg, Pass, Scene, SceneController\n"
        "class D(SceneController):\n"
        "    ARGS = [Arg('emit', 'size'), Arg('emit', 'int')]\n"
        "    def build(self, base):\n"
        "        return Scene(passes=[Pass(name='main', code='void main(){}')])\n"
    )
    with pytest.raises(ValueError, match="--emit"):
        load_scene(str(tmp_path / "dup.py"))


# ------------------------------------------------------------ the CLI ----


def test_split_leaves_gheeflags_alone():
    from ghee.cli import _split_scene_args

    mine, theirs = _split_scene_args(
        ["render", "text", "--size", "64x48", "--", "--text", "hi", "--", "-x"]
    )
    assert mine == ["render", "text", "--size", "64x48"]
    assert theirs == ["--text", "hi", "--", "-x"]
    assert _split_scene_args(["render", "text", "--size", "64x48"]) == (
        ["render", "text", "--size", "64x48"],
        [],
    )


def test_commands_without_scenes_reject_scene_args():
    from ghee.cli import main

    with pytest.raises(SystemExit) as e:
        main(["displays", "--", "--emit", "8x8"])
    assert "no scene arguments" in str(e.value.code)


def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "ghee.cli", *args], capture_output=True, text=True
    )


def test_cli_scene_help_after_its_own_flags():
    r = _cli("render", "--demo", "text", "--size", "64x48", "--", "--help")
    assert r.returncode == 0
    assert "--text TEXT" in r.stdout


def test_cli_own_flags_still_validated():
    r = _cli("render", "text", "--nosuchflag", "--", "--text", "hi")
    assert r.returncode == 2
    assert "unrecognized arguments: --nosuchflag" in r.stderr


def test_serve_help_groups_the_args_by_scene():
    r = _cli("serve", "--demo", "--", "--help")
    assert r.returncode == 0
    assert "text arguments:" in r.stdout
    assert "--text TEXT" in r.stdout
    assert "--emit WxH" in r.stdout


def test_serve_rejects_an_arg_no_scene_declares():
    r = _cli("serve", "--demo", "--", "--emitt", "4x4")
    assert r.returncode == 2
    assert "unrecognized arguments: --emitt" in r.stderr


def test_check_scene_args_over_the_examples(capsys):
    paths = {"text": f"{examples_dir()}/text.py", "bars": f"{examples_dir()}/bars"}
    check_scene_args(paths, ["--text", "hi"], "ghee serve --")
    with pytest.raises(SystemExit):
        check_scene_args(paths, ["--nope"], "ghee serve --")
    assert "unrecognized arguments: --nope" in capsys.readouterr().err


def test_declared_args_without_building(tmp_path):
    assert [a.name for a in declared_args(declaring(tmp_path))] == [
        "emit",
        "word",
        "count",
        "dither",
    ]
    (tmp_path / "flat.scene.json").write_text(JSON_SCENE)
    assert declared_args(str(tmp_path / "flat.scene.json")) == ()


def test_store_hands_its_args_to_every_scene():
    store = SceneStore(examples_dir(), read_only=True, args=["--text", "WWWW"])
    scene, _ = store.resolve("text")
    plain, _ = load_scene(f"{examples_dir()}/text.py")
    assert scene.passes[0].arrays[0].size[0] > plain.passes[0].arrays[0].size[0]
    bars, ctrl = store.resolve("bars")  # takes no args, loads all the same
    assert ctrl is None and bars.passes


# ------------------------------------------------ the bundled examples ----


def test_text_bakes_the_word_it_was_given():
    wide, _ = load_scene(f"{examples_dir()}/text.py", ["--text", "MMMMMM"])
    narrow, _ = load_scene(f"{examples_dir()}/text.py", ["--text", "I"])
    assert wide.passes[0].arrays[0].size[0] > narrow.passes[0].arrays[0].size[0]
    assert wide.passes[0].arrays[0].size[1] == narrow.passes[0].arrays[0].size[1]


def test_text_renders_the_word_it_was_given(renderer):
    def frame_for(args):
        scene, ctrl = load_scene(f"{examples_dir()}/text.py", args)
        assert renderer.setup(scene) == ""
        ctrl.setup(renderer)
        renderer.set_time(0.0)
        return renderer.render(lambda name, size: ctrl.provide(name, size, 0)).copy()

    a, b = frame_for(["--text", "IIII"]), frame_for(["--text", "hello there"])
    assert not np.array_equal(a, b)
    for f in (a, b):
        assert 0.005 < (f[:, :, 1] > 120).mean() < 0.4  # text-like, not a flood


def test_bluesun_emit_sizes_the_heavy_passes():
    plain, _ = load_scene(f"{examples_dir()}/bluesun.py")
    sized, _ = load_scene(f"{examples_dir()}/bluesun.py", ["--emit", "480x300"])
    by_name = {p.name: p for p in plain.passes}
    assert by_name["tube"].size is None  # full render size
    assert by_name["wash"].size == (384, 384)
    by_name = {p.name: p for p in sized.passes}
    assert by_name["tube"].size == (480, 300)
    w, h = by_name["wash"].size  # the wall buffer follows the draw aspect
    assert h == 288 and round(w / h, 2) == 1.6
    forced, _ = load_scene(
        f"{examples_dir()}/bluesun.py", ["--emit", "480x300", "--wash", "64x64"]
    )
    assert {p.name: p.size for p in forced.passes}["wash"] == (64, 64)


def test_hero_grid_and_dither_are_build_time():
    plain, _ = load_scene(f"{examples_dir()}/hero.py")
    assert plain.filters == ["dither"]
    assert {p.name: p.size for p in plain.passes}["glow"] == (320, 200)
    assert "const int COL = 20;" in {p.name: p.code for p in plain.passes}["glow"]
    small, _ = load_scene(
        f"{examples_dir()}/hero.py",
        ["--particle-grid", "6x4", "--particle-buffer", "160x100", "--no-dither"],
    )
    sizes = {p.name: p.size for p in small.passes}
    assert sizes["glow"] == (160, 100)
    assert sizes["state"] == (6, 4)
    code = {p.name: p.code for p in small.passes}["glow"]
    assert "const int COL = 6;" in code and "const int ROW = 4;" in code
    assert small.filters == []


def test_hero_renders_at_a_smaller_grid(renderer):
    scene, ctrl = load_scene(
        f"{examples_dir()}/hero.py",
        ["--particle-grid", "4x3", "--particle-buffer", "64x40", "--no-dither"],
    )
    assert renderer.setup(scene) == ""
    ctrl.setup(renderer)
    frame = None
    for i in range(3):
        ctrl.update(renderer, i, i / 60.0)
        frame = renderer.render(lambda n, s, i=i: ctrl.provide(n, s, i)).copy()
    assert frame.std() > 2  # the word and the water, no dither grid over them
    assert not np.array_equal(frame[..., 0], frame[..., 2])
