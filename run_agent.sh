python -u run_agent.py --use_tech 1 --use_txnstat 0 --use_news 1 --use_reflection 1 &>run_agent-wo-txnstat.out 2>&1
python -u run_agent.py --use_tech 0 --use_txnstat 1 --use_news 1 --use_reflection 1 &>run_agent-wo-tech.out 2>&1
python -u run_agent.py --use_tech 0 --use_txnstat 0 --use_news 0 --use_reflection 0 &>run_agent-base.out 2>&1
python -u run_agent.py --use_tech 1 --use_txnstat 1 --use_news 1 --use_reflection 1 &>run_agent-full.out 2>&1
