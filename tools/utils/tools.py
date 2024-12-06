import os
import tarfile
from io import StringIO, BytesIO
from typing import Dict, Optional, List

import pandas as pd


def find_key_by_list_string(my_dict, str_list):
    for key, value in my_dict.items():
        if set(str_list).issubset(set(value)):
            return key
    return None


def _find_log_files(directory: str) -> List[str]:
    log_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # TODO: this is a temporary solution, any good methods to identify a log file?
            if file.endswith(".csv") or file.endswith(".tar.gz"):
                log_files.append(os.path.join(root, file))
    return log_files

def read_log_file(fn: str) -> pd.DataFrame:
    ext = os.path.splitext(fn)[1]
    compress_method = get_compress_method_from_ext(ext)
    if compress_method is None:
        return pd.read_csv(fn)
    else:
        with open(fn, "rb") as fh:
            data = fh.read()
        data = decompress(data, method=compress_method)
        ret = []
        for k, v in data.items():
            ret.append(pd.read_csv(StringIO(v.decode("utf8"))))
            print("fgfh")
        return pd.concat(ret, axis=0)


def decompress_gz(data: bytes) -> Dict[str, str]:
    ret = {}
    with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                ret[member.name] = tar.extractfile(member).read().decode("utf8")

    return ret


def get_compress_method_from_ext(ext: str) -> Optional[str]:
    return {
        "tar.gz": "gz"
    }.get(ext)


def decompress(data: bytes, method: str = "gz") -> Dict[str, str]:
    if method != "gz":
        raise NotImplementedError("Only 'gz' method is supported by now")
    return decompress_gz(data)


