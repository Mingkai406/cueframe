#!/usr/bin/env python3
"""Fetch checksum-verified assets; optionally build a minimal local OpenCV."""

import argparse, hashlib, pathlib, subprocess, tarfile, urllib.request, os

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_SHA = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
REV = "47534e27c9851bb1128ccc0102f1145e27f23f98"


def fetch(url, path, digest):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        part = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, part)
        if hashlib.sha256(part.read_bytes()).hexdigest() != digest:
            part.unlink()
            raise RuntimeError("Checksum mismatch: " + path.name)
        part.replace(path)
    print("Verified", path.name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--opencv", action="store_true")
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()
    fetch(
        f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{REV}/models/object_detection_yolox/object_detection_yolox_2022nov.onnx",
        ROOT / "models/yolox_s.onnx",
        MODEL_SHA,
    )
    if not a.opencv:
        return
    deps = ROOT / ".deps"
    archive = deps / "opencv-4.12.0.tar.gz"
    fetch(
        "https://github.com/opencv/opencv/archive/refs/tags/4.12.0.tar.gz",
        archive,
        "44c106d5bb47efec04e531fd93008b3fcd1d27138985c5baf4eafac0e1ec9e9d",
    )
    source = deps / "opencv-4.12.0"
    if not source.exists():
        with tarfile.open(archive) as t:
            # The verified archive is also confined to the dependency directory.
            for member in t.getmembers():
                if (
                    not (deps / member.name).resolve().is_relative_to(deps.resolve())
                    or member.issym()
                    or member.islnk()
                ):
                    raise RuntimeError("Unsafe archive entry")
            t.extractall(deps)
    build = deps / "opencv-build"
    install = deps / "opencv-install"
    flags = [
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={install}",
        "-DBUILD_LIST=core,imgproc,imgcodecs,videoio,dnn",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_PERF_TESTS=OFF",
        "-DBUILD_EXAMPLES=OFF",
        "-DBUILD_opencv_apps=OFF",
        "-DBUILD_JAVA=OFF",
        "-DBUILD_opencv_python3=OFF",
        "-DWITH_IPP=OFF",
        "-DWITH_OPENEXR=OFF",
        "-DWITH_OPENCL=OFF",
        "-DWITH_FFMPEG=OFF",
        "-DWITH_GSTREAMER=OFF",
        "-DBUILD_SHARED_LIBS=OFF",
    ]
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), *flags], check=True)
    subprocess.run(["cmake", "--build", str(build), "-j", str(a.jobs)], check=True)
    subprocess.run(["cmake", "--install", str(build)], check=True)
    print("Configure CueFrame with -DCMAKE_PREFIX_PATH=.deps/opencv-install")


if __name__ == "__main__":
    main()
