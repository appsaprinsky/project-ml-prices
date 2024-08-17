import datetime



tickers = ["META", "AAPL", "GOOGL", "AMZN", "MSFT", "TSLA"]
startDate = datetime.datetime(2018, 1, 1)
endDate = datetime.datetime(2022, 1, 1)
TRAIN_TEST_DATE_DIVISION = "2021-01-01"
SELECTED_Y = "TSLA"#"META"

window_size_corr_short = 10
window_size_coint = 5

params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}