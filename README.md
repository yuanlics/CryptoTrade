# example prompt

Play an ETH trading game to maximize your overall profit. At each step, directly suggest an action in precision-two float-value in range [-1,1]. If action in range [-1,0], you will sell the portion of holding ETH at hand. If action in range [0,1], you will spend the portion of cash at hand to buy ETH. If action is 0, no action is done. You start with 10000 dollars and will be asked to act as follows. Remember to only return the precision-two float-value action and nothing more.

Your last action's profit was 0.00 dollars. Your current cash is 10000.00 dollars and holding 0.00 ETH. The ETH open price is 1.35 dollars. Directly give your action in format X.XX >

Your last action's profit was 5314.61 dollars. Your current cash is 7843.53 dollars and holding 54.32 ETH. The ETH open price is 2.64 dollars. Directly return your action in format X.XX >
