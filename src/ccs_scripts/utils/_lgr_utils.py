"""
Low-level utilities for reading Local Grid Refinements (LGR) from Eclipse binary files.

Functions here depend on `resdata` and `resfo`, which are imported lazily to avoid
paying the import cost unless LGR functionality is actually needed.
"""

from pathlib import Path

import resfo


# Keywords in LGR blocks that carry no property data and should be stripped when
# writing the extracted LGR UNRST/INIT to a standalone file.
_SKIP_LGR_HEADERS = {"LGRHEADI", "LGRHEADQ", "LGRHEADD", "LOGIHEAD", "DOUBHEAD"}
_SKIP_WELL_ARRAYS = {
    "IWEL", "SWEL", "XWEL", "ZWEL",
    "IGRP", "SGRP", "XGRP", "ZGRP",
    "ICON", "SCON", "XCON",
    "LGRSGONE", "LGRNAMES",
}


def get_lgr_names(grid_file: Path) -> list[str]:
    """Return the names of all LGRs present in the given EGRID file."""
    from resdata.grid import Grid as ResdataGrid  # lazy import

    rd_grid = ResdataGrid(str(grid_file))
    return [rd_grid.get_lgr(i).get_name() for i in range(rd_grid.get_num_lgr())]


def create_lgr_grid(grid_file: Path, lgr_name: str, lgr_target_file: Path) -> None:
    """Extract a single LGR subgrid from the given EGRID file as an xtgeo Grid.

    The LGR is written to a temporary EGRID file and read back with xtgeo. This is
    simpler and more robust than a direct array conversion between resdata's
    COORD/ZCORN format and xtgeo's coordsv/zcornsv format, which requires non-trivial
    index arithmetic.
    """
    from resdata.grid import Grid as ResdataGrid  # lazy import

    rd_grid = ResdataGrid(str(grid_file))
    lgr = rd_grid.get_lgr(lgr_name)
    # Alternative:
    # import cwrap
    # try:
    #     cf = cwrap.open(str(grid_file), "w")
    #     lgr.save_grdecl(cf, str(lgr_target_file))
    # finally:
    #     cf.close()
    lgr.save_EGRID(str(lgr_target_file))


def extract_lgr_unrst(source_file: Path, lgr_name: str, output_file: Path) -> None:
    """Extract the LGR-specific data for *lgr_name* from *source_file* into
    *output_file*.

    Works with any Eclipse binary restart file (UNRST, INIT, …) that embeds LGR
    blocks using the LGR/ENDLGR keyword convention.  The output is a valid file that
    xtgeo can read as an ordinary UNRST/INIT: LGR wrapper headers are stripped, but the
    LGR's own INTEHEAD (which carries the correct grid dimensions and date) is
    preserved.
    """
    seqnum = None
    in_target_lgr = False
    block_records: list[tuple[str, object]] = []
    all_blocks: list[list[tuple[str, object]]] = []

    for entry in resfo.lazy_read(source_file):
        kw = entry.read_keyword().rstrip()

        if kw == "SEQNUM":
            seqnum = entry.read_array()
            in_target_lgr = False
        elif kw == "LGR":
            arr = entry.read_array()
            decoded = (
                b"".join(arr).decode("ascii", errors="replace").strip()
                if arr.dtype.kind == "S"
                else "".join(arr).strip()
            )
            if decoded == lgr_name:
                in_target_lgr = True
                block_records = [("SEQNUM  ", seqnum)]
        elif kw == "ENDLGR":
            if in_target_lgr:
                all_blocks.append(block_records)
            in_target_lgr = False
        elif in_target_lgr:
            if kw in _SKIP_LGR_HEADERS or kw in _SKIP_WELL_ARRAYS:
                continue
            block_records.append((f"{kw:<8}", entry.read_array()))

    records = [rec for block in all_blocks for rec in block]
    resfo.write(output_file, records)
