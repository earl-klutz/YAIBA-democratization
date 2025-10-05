from src.yaiba_bi import core


def main():
    path = "../test_log.txt"
    log_data: core.LogData = core.load(path)
    area = log_data.get_area()

    gen = core.HeatmapGenerator(boundary=area)

    gen.run(
        log_data.get_position(),
        output_basename="testdata",
        save_dir="../"
    )


if __name__ == "__main__":
    main()
