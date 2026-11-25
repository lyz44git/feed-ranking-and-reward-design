"""Keep the captions of the pool's videos from KuaiRand's caption file (3.2 GB, all 32 million videos).

Usage: python steps/s4_extract_captions.py <kuairand_video_captions.csv> <KuaiRand-Pure/data> <out.parquet>
"""
import sys
from pathlib import Path

import duckdb


def main():
    src, data, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    c = duckdb.connect()
    c.sql(f"""copy (select k.* from read_csv('{src}', header=true, ignore_errors=true,
                     columns={{'final_video_id': 'BIGINT', 'caption': 'VARCHAR', 'show_cover_text': 'VARCHAR', 'duration': 'DOUBLE'}}) k
                    semi join read_csv_auto('{data / 'video_features_basic_pure.csv'}') v on k.final_video_id = v.video_id)
              to '{out}' (format parquet)""")
    print(c.sql(f"""select count(*) as videos, avg((abs(k.duration - v.video_duration) < 1000)::int) as length_matches
                    from '{out}' k join read_csv_auto('{data / 'video_features_basic_pure.csv'}') v
                    on k.final_video_id = v.video_id""").fetchall())


if __name__ == "__main__":
    main()
