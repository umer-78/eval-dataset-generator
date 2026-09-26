3,080 logged messages; 400 near-duplicates dropped; 71 clusters found (largest: transfer / long / pending; exchange / currencies / currency; reverted / pending / working; identity / verify / check; atm / cash / didn), 986 outliers. The production model is right on 92.6%. Failure rate among outliers 12.6%, elsewhere 4.3%.

Labelling budget 200 cases, 300 draws per strategy (means):

| Strategy | Cases | Model failures captured | Intents covered (of 77) | Clusters covered | Accuracy estimate RMSE | Bias, weighted | Bias, unweighted mean |
|---|---|---|---|---|---|---|---|
| uniform | 200 | 14.8 | 71.6 | 49.8 | 1.98 pts | -0.05 pts | -0.05 pts |
| stratified | 200 | 14.0 | 71.5 | 53.8 | 1.89 pts | +0.03 pts | +0.38 pts |
| boosted | 200 | 25.6 | 70.2 | 37.9 | 1.43 pts | -0.16 pts | -5.52 pts |

A built set (boosted, version f7d919c2590b), first cases, with the gold label the labeller would add (Banking77, CC BY 4.0):

| Message | Production answer | Gold | Cluster | Confidence | Weight |
|---|---|---|---|---|---|
| I ordered a card but it has not arrived. Help please! | card_arrival | card_arrival | arrived / ordered / card | 0.744 | 16.38 |
| Is there a way to know when my card will arrive? | card_arrival | card_arrival | outlier | 0.758 | 8.37 |
| How long does a card delivery take? | card_delivery_estimate | card_arrival | outlier | 0.884 | 9.58 |
| still waiting on my new card | card_arrival | card_arrival | arrive / card / long | 0.938 | 26.62 |
| I did not get my card yet, is it lost? | card_arrival | card_arrival | outlier | 0.789 | 8.63 |
| Status of the card I ordered. | card_arrival | card_arrival | outlier | 0.566 | 7.01 |
