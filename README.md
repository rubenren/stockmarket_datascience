# production_repo
Production level code for Stock Market Prediction

How to use code:
1) First add the stock you want with `python add --stock [STOCK]`
2) Then you need to update the database with recent stock data using `python main.py download`
3) After having downloading the datasets, you may run `python main.py predict --stock [STOCK]`
4) The results will be stored in result.txt, either it goes up or down

 - You can also remove a stock from being tracked by using `python main.py remove --stock [STOCK]`