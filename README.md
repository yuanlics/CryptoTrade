# prompt for news formatter.

Summarize a financial news using elements including Who, What, When, Where, Why, How, Sentiment. The article is given below:
Title: {item["title"]}.
Content: {item["content"]}.
Briefly summarize the news in the aspect of {aspect}.

```
example news: 
"id": 29,
"time": "Mon, 01 Jan 2024 08:00:00 GMT",
"title": "Ethereum Price Could Start 2024 With A Strong Increase If It Holds This Support - NewsBTC",

llama13b-fp16
[
    {
        "id": 29,
        "title": "Ethereum Price Could Start 2024 With A Strong Increase If It Holds This Support - NewsBTC",
        "summary": "  Sure! Here is a summary of the financial news article you provided:\n\nEthereum's price is currently correcting gains below the $2,350 level and is trading below the 100-hourly Simple Moving Average. If the price holds above the $2,350 support, it could attempt a fresh increase and potentially reach $2,400. However, if Ethereum fails to clear the $2,300 resistance, it could continue to move down and test the $2,240 and $2,200 support levels. The technical indicators show that the MACD is losing momentum in the bearish zone, and the RSI is below the 50 level. The key resistance levels are $2,350 and $2,440, and the key support levels are $2,240 and $2,200."
    }
]

llama7b-fp32
[
    {
        "id": 29,
        "title": "Ethereum Price Could Start 2024 With A Strong Increase If It Holds This Support - NewsBTC",
        "summary": "  The article discusses the current price action of Ethereum (ETH) and its potential for future movement. According to the analysis, ETH is correcting gains and trading below the $2,350 level, with a key bearish trend line forming near $2,300. However, if the price can close above $2,350, it could indicate a fresh increase towards $2,400 and potentially even $2,500. On the other hand, if ETH fails to hold above $2,300, it could lead to further losses towards $2,240 and potentially even $2,120.\n\nThe article also provides technical indicators such as the MACD and RSI, which are losing momentum in the bearish zone. The major support levels are $2,240 and $2,200, while the major resistance levels are $2,350 and $2,440.\n\nOverall, the article suggests that the future movement of Ethereum is uncertain and will depend on the price's ability to hold above $2,350. If the price can close above this level, it could indicate a fresh increase, but if it fails to hold above $2,300, it could lead to further losses."
    }
]

llama7b-fp16
[
    {
        "id": 29,
        "title": "Ethereum Price Could Start 2024 With A Strong Increase If It Holds This Support - NewsBTC",
        "summary": "  The article discusses the potential price movement of Ethereum (ETH) in the coming days, based on its current technical analysis. According to the analysis, ETH is correcting gains and trading below the $2,350 level, which could lead to a fresh increase in price if the currency manages to close above $2,350. However, if ETH fails to clear the $2,300 resistance, it could continue to move down, potentially reaching $2,240 or lower.\n\nThe technical indicators mentioned in the article include the MACD (Moving Average Convergence Divergence) and the RSI (Relative Strength Index), which are both losing momentum in the bearish zone. The major support levels mentioned are $2,240 and $2,120, while the major resistance levels are $2,350, $2,400, and $2,500.\n\nOverall, the article suggests that the price of Ethereum could potentially start a fresh increase if it can close above $2,350, but if it fails to do so, it could continue to move down and"
    }
]
```


# prompt for trading bot.

Model: gpt3.5

---

Play an ETH trading game to maximize your overall profit. At each step, directly suggest an action in precision-two float-value in range [-1,1]. If action in range [-1,0], you will sell the portion of holding ETH at hand. If action in range [0,1], you will spend the portion of cash at hand to buy ETH. If action is 0, no action is done. You start with 10000 dollars and will be asked to act as follows. Remember to only return the precision-two float-value action and nothing more.

Your last action's profit was 0.00 dollars. Your current cash is 10000.00 dollars and holding 0.00 ETH. The ETH open price is 1.35 dollars. Directly give your action in format X.XX >

Your last action's profit was 5314.61 dollars. Your current cash is 7843.53 dollars and holding 54.32 ETH. The ETH open price is 2.64 dollars. Directly return your action in format X.XX >
