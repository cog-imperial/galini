# Copyright 2018 Francesco Ceccon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate and format output."""
from io import StringIO
import json
import sys
from texttable import Texttable


class OutputTable(object):
    def __init__(self, name, columns):
        self.name = name
        self.rows = []

        if len(columns) == 0:
            raise ValueError('OutputTable must contain at least one column.')

        column = columns[0]
        if isinstance(column, str):
            # simple column format. no types, no header
            self.columns_id = columns
            self.columns_name = columns
            self.columns_type = None
        elif isinstance(column, dict):
            # column format with name and type
            self.columns_id = [column['id'] for column in columns]
            self.columns_name = [column['name'] for column in columns]
            self.columns_type = [column.get('type', 't') for column in columns]
        else:
            raise ValueError('OutputTable columns in wrong format.')

    def add_row(self, row):
        self.rows.append(row)


class JsonFormatter(object):
    pass


class TextFormatter(object):
    pass


def add_output_format_parser_arguments(parser):
    parser.add_argument(
        '--output',
        dest='output',
        default=None
    )
    parser.add_argument(
        '--format',
        dest='output_format',
        default='text',
        choices=['text', 'json']
    )


def print_output_table(table, args):
    if args.output is None:
        out_file = sys.stdout
    else:
        out_file = open(args.output, 'w')
    if args.output_format == 'text':
        _print_output_table_as_text(table, out_file)
    elif args.output_format == 'json':
        _print_output_table_as_json(table, out_file)
    else:
        raise RuntimeError('Invalid output_format {}'.format(args.output_format))
    if args.output is not None:
        out_file.close()


def _print_output_table_as_text(table, out_file):
    out = StringIO()
    if isinstance(table, list):
        for t in table:
            _write_output_table(out, t, write_table_name=True)
            out.write('\n')
    else:
        _write_output_table(out, table)
    out_file.write(out.getvalue())


def _print_output_table_as_json(table, out_file):
    if not isinstance(table, list):
        tables = [table]
    else:
        tables = table

    output = dict()

    for table in tables:
        output[table.name] = _output_table_as_json(table)
    out_file.write(json.dumps(output))


def _write_output_table(writer, table, write_table_name=False):
    tt = Texttable()
    tt.set_deco(Texttable.HEADER)
    tt.set_precision(10)

    if table.columns_type:
        tt.set_cols_dtype(table.columns_type)
    tt.header(table.columns_name)
    ids = table.columns_id
    for raw_row in table.rows:
        row = [raw_row[i] for i in ids]
        tt.add_row(row)

    drawn_table = tt.draw()

    writer.write('\n')

    if write_table_name:
        line_length = len(drawn_table.split('\n')[0])
        spacing = (line_length - len(table.name)) // 2

        writer.write(' ' * spacing)
        writer.write(table.name)
        writer.write('\n')

    writer.write(tt.draw())


def _output_table_as_json(table):
    return [row for row in table.rows]
