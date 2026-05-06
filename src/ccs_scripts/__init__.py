# import logging
# import sys
#
# try:
#     import pkg_resources
#
#     __version__ = pkg_resources.get_distribution(__name__).version
# except pkg_resources.DistributionNotFound:
#     pass
#
#
# def getLogger(module_name="ccs-scripts"):
#     # pylint: disable=invalid-name
#     """Provides a unified logger for subscript scripts.
#
#     Scripts in subscript are encouraged to use logging.info() instead of
#     print().
#
#     The logger name will typically be "subscript.prtvol2csv" for the command
#     line tool prtvol2csv.
#
#     Subscript scripts can set the level of the entire logger, through the
#     setLevel() function. The default level is WARNING. Subscript scripts
#     typically accept a --verbose argparse option to set the log level to INFO,
#     and a --debug option to set to
#
#     Logging output is split by logging levels (split between WARNING and ERROR)
#     to stdout and stderr, each log occurs in only one of the streams. This
#     deviates from Unix standard, but is accepted here because few to none
#     subscript tool are meant to have their stdout piped into another
#     application by default (some of them can, then the programmer and user must
#     be careful with log levels).
#
#     Args:
#         module_name (str): A suggested name for the logger, usually
#             __name__ should be supplied
#
#     Returns:
#         A logger object
#     """
#     if not module_name:
#         return getLogger("ccs-scripts")
#
#     # This logger is also used by subscript-internal, but we
#     # don't want to expose that detail and repo difference in
#     # the log output:
#     module_name = module_name.replace("subscript_internal", "subscript")
#
#     compressed_name = []
#     for elem in module_name.split("."):
#         if len(compressed_name) == 0 or elem != compressed_name[-1]:
#             compressed_name.append(elem)
#
#     logger = logging.getLogger(".".join(compressed_name))
#
#     formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
#
#     stdout_handler = logging.StreamHandler(sys.stdout)
#     stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)
#     stdout_handler.setFormatter(formatter)
#
#     stderr_handler = logging.StreamHandler(sys.stderr)
#     stderr_handler.addFilter(lambda record: record.levelno >= logging.ERROR)
#     stderr_handler.setFormatter(formatter)
#
#     logger.addHandler(stdout_handler)
#     logger.addHandler(stderr_handler)
#
#     return logger
#

def _monkey_patch_scan_dates() -> None:
    # Monkey-patch function
    # Instead of reading data for each iteration, data is only read
    # when it is used.
    import pandas as pd
    import resfo

    print("Applying monkey-patch to xtgeo's scan_dates function")

    def scan_dates_patch(
        pfile,
        maxdates: int,  # Dropped MAXDATES default since it is not needed in the patch
        dataframe: bool = False,
    ) -> list | pd.DataFrame:
        """Quick scan dates in a simulation restart file.

        Cf. grid_properties.py description
        """
        print("Invoking patched scan_dates function")
        dates = []
        seqnum = -1
        for item in resfo.lazy_read(pfile.file):
            kw = item.read_keyword().strip()

            data = None
            if kw == "SEQNUM":
                data = item.read_array()
                seqnum = data[0]
                continue

            # With LGRs multiple INTEHEADs may occur. Ensure we get the date
            # from the first INTEHEAD after a SEQNUM.
            if kw == "INTEHEAD" and seqnum != -1:
                if data is None:
                    data = item.read_array()
                # Index 66 = year, 65 = month, 64 = day
                date = int(f"{data[66]}{data[65]:02d}{data[64]:02d}")
                dates.append((seqnum, date))
                seqnum = -1

        return (
            pd.DataFrame.from_records(dates, columns=["SEQNUM", "DATE"])
            if dataframe
            else dates
        )

    from xtgeo.grid3d import _grid3d_utils
    _grid3d_utils.scan_dates = scan_dates_patch


_monkey_patch_scan_dates()
