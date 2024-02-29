# prompt for news formatter.

Summarize a financial news using elements including Who, What, When, Where, Why, How, Sentiment. The article is given below:
Title: {item["title"]}.
Content: {item["content"]}.
Briefly summarize the news in the aspect of {aspect}.

```
llama-13b-fp16 example
[
    {
        "Who": "  Sure! Here's a summary of the news in the aspect of Who:\n\nWho:\n\n* Luke Nolan, CoinShares Ethereum research associate\n* The Ethereum network\n* Layer 2 transactions\n\nWho might be affected:\n\n* Ethereum users who rely on Layer 2 transactions\n* Investors who hold ether and are affected by the potential change in gas prices and ether burn mechanism.",
        "What": "  Sure! Here's a brief summary of the news in the aspect of \"What\":\n\nThe Ethereum network's upcoming Dencun upgrade could impact the supply of ether by reducing gas usage and, consequently, the amount of ether burned. The upgrade will enable Layer 2 transactions to use a new blobspace mechanism, which has significantly lower gas costs than the current transactional calldata mechanism. This could lead to lower gas prices and less ether being burned, potentially impacting the growth of the ether supply.",
        "When": "  Sure! Here's a summary of the news in the aspect of When:\n\nWhen:\n\n* The Dencun upgrade was activated on the Goerli testnet on Jan. 17.\n* The Ethereum Dencun upgrade on the Sepolia testnet is expected to be activated by Jan. 30.\n* The same upgrade on the Holesky testnet remains scheduled for activation on Feb. 7.\n* The mainnet activation of the Dencun upgrade could come as soon as March of this year.",
        "Where": "  Sure! Here's a summary of the news in the aspect of Where:\n\nWhere: The Dencun upgrade is currently being tested on Ethereum's testnets, including Goerli, Sepolia, and Holesky. The mainnet activation of the upgrade is expected to happen in March.\n\nHere's a breakdown of the summary using the Who, What, When, Where, Why, and How elements:\n\nWho: Luke Nolan, CoinShares Ethereum research associate\n\nWhat: The Dencun upgrade, which enables the blobspace mechanism for Layer 2 transactions\n\nWhen: Currently being tested on testnets, with mainnet activation expected in March\n\nWhere: Ethereum's testnets (Goerli, Sepolia, and Holesky) and mainnet\n\nWhy: To reduce gas usage and lower the amount of ether burned, potentially impacting the growth of ether supply\n\nHow: By using the new blobspace mechanism instead of transactional calldata, which has significantly lower gas costs",
        "Why": "  Sure! Here's the summary of the news in the aspect of \"Why\":\n\nThe Dencun upgrade for the Ethereum network could potentially reduce the amount of ether burned, as it enables Layer 2 transactions to use a new blobspace mechanism that has significantly lower gas costs. This could lead to lower gas prices and a decrease in the circulating supply of ether, which could impact the growth of the ether supply. However, the impact of the upgrade could be mitigated by other factors such as Layer 1 demand, and the end goal of the upgrade is to solidify Ethereum's market share.",
        "How": "  Sure! Here's a summary of the news in the aspect of How:\n\nThe Dencun upgrade for the Ethereum network is expected to be activated on the mainnet in March, with the goal of reducing gas fees and increasing network activity. The upgrade will enable the use of the blobspace mechanism, which has lower gas costs and could potentially reduce the amount of ether burned. However, the impact of the upgrade on the ether supply is uncertain and may be mitigated by other factors.",
        "Sentiment": "  Sure! Here's a summary of the news in the aspect of Sentiment:\n\nSentiment: Neutral\n\nThe article discusses the potential impact of Ethereum's Dencun upgrade on the supply of ether. While the upgrade could reduce the amount of ether burned, it is not expected to have a significant impact on the growth of ether supply. The upgrade is aimed at decreasing gas fees for users, which could lead to more activity on the network and potentially more gas usage overall. The sentiment is neutral, as the upgrade is seen as a positive development for the network, but the impact on ether supply is not expected to be significant."
    }
]

llama-7b-fp16 example
[
    {
        "Who": "  Sure, here's a summary of the news in the aspect of \"Who\":\n\nWho: Luke Nolan, CoinShares Ethereum research associate\n\nLuke Nolan is the person who provided insights into the potential impact of the Ethereum Dencun upgrade on the supply of ether. He is a research associate at CoinShares, a leading blockchain research and investment firm. Nolan has a deep understanding of the Ethereum network and its mechanics, and his comments provide valuable perspective on the potential effects of the Dencun upgrade.",
        "What": "  The article discusses the potential impact of the Ethereum network's upcoming Dencun upgrade on the supply of ether. According to CoinShares Ethereum research associate Luke Nolan, the upgrade will enable Layer 2 transactions to be sent to the Ethereum network through blobspace, an alternative to the current transactional calldata mechanism. This could result in reduced gas usage and, consequently, a lower amount of ether being burned. The possible reduction in ether burned could have an impact on the growth of ether supply. However, Nolan stresses that Layer 1 demand is what significantly drives deflation, and that even if gas prices settle at lower levels, there is no real concern of Ethereum becoming significantly inflationary.",
        "When": "  Sure! Here is a summary of the news in the aspect of \"When\":\n\n* The Dencun upgrade was activated on the Goerli testnet on January 17.\n* The Dencun upgrade on the Sepolia testnet is expected to be activated by January 30.\n* The same upgrade on the Holesky testnet remains scheduled for activation on February 7.\n* The mainnet activation of the Dencun upgrade could come as soon as March of this year.",
        "Where": "  Sure! Here's a summary of the news in the aspect of \"Where\":\n\nThe Ethereum Dencun upgrade is expected to have an impact on the supply of ether, as the new blobspace mechanism could result in reduced gas usage and, consequently, a lower amount of ether being burned. The upgrade is currently being tested on various testnets, including Goerli, Sepolia, and Holesky. The mainnet activation of the Dencun upgrade is expected to happen sometime in March of this year, although the exact timeline is subject to change as testnet bugs appear and are addressed.",
        "Why": "  The news article discusses the potential impact of the Ethereum network's Dencun upgrade on the supply of ether. According to CoinShares Ethereum research associate Luke Nolan, the upgrade will enable Layer 2 transactions to be sent to the Ethereum network through blobspace, which could result in reduced gas usage and, consequently, a lower amount of ether being burned. This could have an impact on the growth of ether supply, as gas demand on the Ethereum network drives ether's deflation mechanism via gas burn, which can reduce the circulating supply of ether. However, Nolan notes that even if gas prices settle at lower levels, there is no real concern that Ethereum will become significantly inflationary, as the end goal of the Dencun upgrade is to solidify Ethereum's market share.",
        "How": "  The article discusses how the Ethereum network's upcoming Dencun upgrade could potentially impact the supply of ether. The upgrade enables Layer 2 transactions to be sent to the Ethereum network through blobspace, an alternative to the current transactional calldata mechanism. This could result in reduced gas usage and, consequently, a lower amount of ether being burned. The impact on the ether burn mechanism could have an effect on the growth of ether supply. However, other factors such as Layer 1 demand driving deflation could mitigate the impact of the Dencun upgrade. The end goal of the upgrade is to solidify Ethereum's market share, and any secondary effects are considered a net positive over the long run. The mainnet activation of the Dencun upgrade could come as soon as March of this year.",
        "Sentiment": "  The sentiment of the news article is neutral. While the Dencun upgrade may potentially reduce the amount of ether burned, the impact on the ether supply growth is uncertain and may be mitigated by other factors. The end goal of the upgrade is to solidify Ethereum's market share, and any short-term gas fluctuations are seen as a net positive over the long run. The price of ether has fallen in the past 24 hours, but the article does not provide a clear reason for this movement. Overall, the tone of the article is informative and objective, without expressing a clear positive or negative sentiment towards the news."
    }
]

gemma-2b-fp16 example
[
    {
        "Who": "Sure, here's a brief summary of the news in the aspect of Who:\n\n* Luke Nolan is a CoinShares Ethereum research associate.",
        "What": "Sure, here's a brief summary of the news in the aspect of What:\n\nThe Ethereum Dencun upgrade will enable Layer 2 transactions to be sent to the Ethereum network through blobspace, an alternative to the current transactional calldata mechanism. This could result in reduced gas usage and, consequently, a lower amount of ether being burned.",
        "When": "Sure, here's a brief summary of the news in the aspect of When:\n\nThe Ethereum Dencun upgrade is expected to be activated on the Goerli testnet on Jan. 17, and on the Sepolia and Holesky testnets on Jan. 30 and Feb. 7, respectively.",
        "Where": "The news is about the Ethereum Dencun upgrade, a network upgrade that will enable Layer 2 transactions to be sent to the Ethereum network through blobspace, an alternative to the current transactional calldata mechanism.",
        "Why": "Sure, here's a brief summary of the news in the aspect of Why:\n\nThe Dencun upgrade could potentially reduce gas fees for users transacting on the Ethereum network. This could lead to a lower amount of ether being burned, which could have a positive impact on the growth of ether supply.",
        "How": "Sure, here's a brief summary of the news in the aspect of How:\n\n* The Dencun upgrade will enable Layer 2 transactions to be sent to the Ethereum network through blobspace, an alternative to the current transactional calldata mechanism.\n* This could result in reduced gas usage and, consequently, a lower amount of ether being burned.\n* The Dencun upgrade could see gas prices settle at lower levels, which means less ether is burned.",
        "Sentiment": "The news has a negative sentiment, as it discusses the potential impact of the Ethereum Dencun upgrade on the ether supply and deflation mechanism. The upgrade could potentially lead to lower gas fees and a reduction in ether burned, which could have a negative impact on the growth of ether supply."
    }
]
```


# prompt for trading bot.

Model: gpt3.5

---

Play an ETH trading game to maximize your overall profit. At each step, directly suggest an action in precision-two float-value in range [-1,1]. If action in range [-1,0], you will sell the portion of holding ETH at hand. If action in range [0,1], you will spend the portion of cash at hand to buy ETH. If action is 0, no action is done. You start with 10000 dollars and will be asked to act as follows. Remember to only return the precision-two float-value action and nothing more.

Your last action's profit was 0.00 dollars. Your current cash is 10000.00 dollars and holding 0.00 ETH. The ETH open price is 1.35 dollars. Directly give your action in format X.XX >

Your last action's profit was 5314.61 dollars. Your current cash is 7843.53 dollars and holding 54.32 ETH. The ETH open price is 2.64 dollars. Directly return your action in format X.XX >
