"""
Module containing text column operations that act on geometry blocks
"""
import pandas as pd
import re

from .base import GeometryBlock, BaseSingle

__all__ = ["ParseTextColumn"]

# https://catonmat.net/my-favorite-regex
# key matches any ASCII char except for '='
# value matches any ASCII char
REGEX_KEYVALUE = re.compile(r"((?:[ -<>-~])+)=((?:[ -~])*)")


def autocast_value(value):
    """Cast string to string, float, bool or None. """
    if value is None:
        return
    value_lcase = value.lower()
    if value_lcase == "null":
        return
    if value_lcase == "false":
        return False
    if value_lcase == "true":
        return True
    try:
        return float(value)
    except ValueError:
        return value


class ParseTextColumn(BaseSingle):
    """Parses a text column into (possibly multiple) value columns.

    Key, value pairs need to be separated by an equal (``=``) sign.

    :param source: data source
    :param source_column: existing column in source.
    :param key_mapping: mapping containing pairs {key_name: column_name}:
        key_name: existing key in the text to be parsed.
        column_name: name of the new column created that contains
        the parsed value.

    :type source: GeometryBlock
    :type source_column: string
    :type key_mapping: dict
    """

    def __init__(self, source, source_column, key_mapping):
        if not isinstance(source, GeometryBlock):
            raise TypeError("'{}' object is not allowed.".format(type(source)))
        if not isinstance(source_column, str):
            raise TypeError("'{}' object is not allowed.".format(type(source_column)))
        if source_column not in source.columns:
            raise KeyError("Column '{}' is not available.".format(source_column))
        if not isinstance(key_mapping, dict):
            raise TypeError("'{}' object is not allowed.".format(type(key_mapping)))
        super().__init__(source, source_column, key_mapping)

    @property
    def source(self):
        return self.args[0]

    @property
    def source_column(self):
        return self.args[1]

    @property
    def key_mapping(self):
        return self.args[2]

    @property
    def columns(self):
        return self.source.columns | set(self.key_mapping.values())

    def get_sources_and_requests(self, **request):
        process_kwargs = {
            "source_column": self.source_column,
            "key_mapping": self.key_mapping,
        }
        return [(self.source, request), (process_kwargs, None)]

    @staticmethod
    def process(data, kwargs):
        source_column = kwargs["source_column"]
        key_mapping = kwargs["key_mapping"]

        if "features" not in data or len(data["features"]) == 0:
            return data  # do nothing for non-feature requests

        f = data["features"].copy()
        # Most of the code in this function revolves around the idea that we do
        # not keep large strings multiple times in memory. We use the pandas
        # categorical for that.

        # Hopefully, this is already the case:
        column = f[source_column].astype("category")

        if len(column.cat.categories) == 0:
            # no data to parse: add empty columns and return directly
            f = f.reindex(columns=list(f.columns) + list(key_mapping.values()))
            return {"features": f, "projection": data["projection"]}

        def parser(description):
            pairs = dict(REGEX_KEYVALUE.findall(description))
            return [autocast_value(pairs.get(key)) for key in key_mapping.keys()]

        # Parse each category
        extra_columns = pd.DataFrame(
            [parser(x) for x in column.cat.categories], columns=key_mapping.values()
        )

        # Convert strings to categoricals before aligning
        for name, dtype in zip(extra_columns.columns, extra_columns.dtypes):
            if dtype == object:
                extra_columns[name] = extra_columns[name].astype("category")

        # Align the generated dataframe with the original. Pandas versions
        # later than 0.19 have a pd.align that could be used also.
        try:
            extra_columns_aligned = extra_columns.loc[column.cat.codes]
            extra_columns_aligned.index = f.index
        except KeyError:
            extra_columns_aligned = pd.DataFrame([], columns=key_mapping.values())

        # Assign the extra columns one by one. This preserves categoricals.
        for name in extra_columns_aligned.columns:
            f[name] = extra_columns_aligned[name]

        return {"features": f, "projection": data["projection"]}
