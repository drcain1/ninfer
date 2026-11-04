"""Write a Volta-packed v3 artifact without modifying its represented weights."""
from __future__ import annotations

import argparse
from pathlib import Path

from .codecs.volta import pack_planes, unpack_block_scales
from .layouts import block_scale_geometry, row_scale_geometry
from .reader import Artifact
from .schema import ResourceSpec, TensorObject, TensorSpec
from .writer import ArtifactWriter


def pack(source: Path, destination: Path) -> None:
    with Artifact(source) as artifact:
        specs = []
        selected = set()
        for obj in artifact.objects:
            if isinstance(obj, TensorObject):
                layout = obj.layout
                if layout in ("row_scale_v1", "block_scale_k16_m128x4_v1"):
                    layout = "volta_qpn_v1"
                    selected.add(obj.id)
                specs.append(TensorSpec(obj.id, obj.shape, obj.format, layout))
            else:
                specs.append(ResourceSpec(obj.id, obj.bytes, obj.encoding))
        d = artifact.directory
        with ArtifactWriter(destination, specs, components=d.components, bindings=d.bindings,
                            uses=d.uses, metadata=d.metadata,
                            provenance={**d.provenance, "volta_packing": {"source": str(source),
                                        "source_artifact_id": artifact.artifact_id.hex()}}) as writer:
            for obj in artifact.objects:
                if obj.id not in selected:
                    writer.write_object(obj.id, artifact.iter_object(obj.id))
                    continue
                n, k = obj.shape
                nvfp4 = obj.format == "nvfp4"
                g = (block_scale_geometry if nvfp4 else row_scale_geometry)(obj.format, obj.shape)
                code_row, scale_row = (k // 2, k // 16) if nvfp4 else (k, 2)
                # Complete M128 tiles keep both source and destination ranges contiguous.
                for start in range(0, n, 128):
                    rows = min(128, n - start)
                    codes = artifact.read_range(obj.offset + start * code_row, rows * code_row)
                    scales = artifact.read_range(obj.offset + g.scale_plane_offset + start * scale_row,
                                                 rows * scale_row)
                    if nvfp4:
                        scales = unpack_block_scales(scales, (rows, k))
                    codes, scales = pack_planes(codes, scales, obj.format, (rows, k))
                    writer.write_region(obj.id, start * code_row, codes)
                    writer.write_region(obj.id, g.scale_plane_offset + start * scale_row, scales)
                if nvfp4:
                    writer.write_region(obj.id, g.weight_divisor_offset,
                                        artifact.read_range(obj.offset + g.weight_divisor_offset, 4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    pack(args.source, args.destination)


if __name__ == "__main__":
    main()
