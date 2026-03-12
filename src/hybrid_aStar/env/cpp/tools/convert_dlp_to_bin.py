import argparse
import os
import pickle
import struct


def _as_list(value):
    if isinstance(value, tuple):
        return list(value)
    return value


def _extract_coords(obstacle):
    if hasattr(obstacle, "coords"):
        return list(obstacle.coords)
    if hasattr(obstacle, "exterior"):
        return list(obstacle.exterior.coords)
    raise ValueError("Unsupported obstacle type")


def main():
    parser = argparse.ArgumentParser(description="Convert dlp.data (pickle) to a binary file for C++.")
    parser.add_argument("--input", required=True, help="Path to dlp.data (pickle)")
    parser.add_argument("--output", required=True, help="Output path for dlp.bin")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(struct.pack("<I", len(data)))
        for entry in data:
            start, dest, obstacles = entry[:3]
            start = _as_list(start)
            dest = _as_list(dest)

            if isinstance(start, list) and start and isinstance(start[0], (list, tuple)):
                starts = start
            else:
                starts = [start]

            f.write(struct.pack("<I", len(starts)))
            for s in starts:
                s = _as_list(s)
                f.write(struct.pack("<3d", float(s[0]), float(s[1]), float(s[2])))

            f.write(struct.pack("<3d", float(dest[0]), float(dest[1]), float(dest[2])))

            f.write(struct.pack("<I", len(obstacles)))
            for obs in obstacles:
                coords = _extract_coords(obs)
                f.write(struct.pack("<I", len(coords)))
                for x, y in coords:
                    f.write(struct.pack("<2d", float(x), float(y)))

    print("saved:", args.output)


if __name__ == "__main__":
    main()
