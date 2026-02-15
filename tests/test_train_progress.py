from train_progress import TrainingProgressReporter


def test_progress_reporter_emits_every_n_games():
    reporter = TrainingProgressReporter(
        epoch=2,
        total_epochs=10,
        log_every_games=100,
        start_time=10.0,
        time_fn=lambda: 110.5,
    )

    assert (
        reporter.maybe_report(
            file_index=3,
            file_count=256,
            total_games=99,
            avg_loss=0.5,
            last_loss=0.4,
        )
        is None
    )

    message = reporter.maybe_report(
        file_index=3,
        file_count=256,
        total_games=100,
        avg_loss=0.5,
        last_loss=0.4,
    )
    assert (
        message
        == "progress epoch=2/10 file=3/256 games=100 avg_loss=0.500000 loss=0.400000 elapsed=100.5s"
    )


def test_progress_reporter_disables_when_non_positive_interval():
    reporter = TrainingProgressReporter(
        epoch=1,
        total_epochs=10,
        log_every_games=0,
        start_time=10.0,
        time_fn=lambda: 20.0,
    )
    assert (
        reporter.maybe_report(
            file_index=1,
            file_count=256,
            total_games=100,
            avg_loss=0.1,
            last_loss=0.1,
        )
        is None
    )
