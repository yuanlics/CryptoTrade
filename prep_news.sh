nohup python -u prep_news.py --gpu 5 --start_ymd 2024-01-01 --end_ymd 2024-01-10 &>prep_news1.out 2>&1 &
nohup python -u prep_news.py --gpu 6 --start_ymd 2024-01-11 --end_ymd 2024-01-20 &>prep_news2.out 2>&1 &
nohup python -u prep_news.py --gpu 7 --start_ymd 2024-01-21 --end_ymd 2024-01-30 &>prep_news3.out 2>&1 &
