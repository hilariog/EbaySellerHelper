# EbaySellerHelper
A project using Traject Data's Countdown API to analyze ebay data for the benefit of sellers.

Implementation Summary:

This code provides a compact, data-driven way for eBay sellers to understand their listing performance and discover what factors drive more successful sales. By extracting real-time data (in minimal sample form to preserve API quotas) and generating a robust set of metrics—like market quality, condition-based success, shipping effects, and seller reputation—it reveals how auction vs. buy now formats compare and how attributes such as image count, reviews, and brand relate to final prices and success rates. The “so-what” here is that businesses or individual sellers can use these outputs (charts, correlations, histograms, and summary statistics) to refine their listings—for example, adjusting the number of pictures, choosing an auction vs. fixed price, or focusing on keywords and brand attributes that historically yield better outcomes. Ultimately, this tool offers an evidence-based framework for improving listing strategies and maximizing sales potential on eBay.

Actual Implementation Notes:

-Fields that can be identified for the items (this is not the exact same as my original plan) 
Information that gets saved for every listing:

item_data = {
            "title": title,
            "is_sold": is_sold,
            "is_auction": is_auction,
            "buy_it_now": buy_it_now,
            "final_price": final_price,
            "ended_dt": ended_dt,
            "image_count": image_count,
            "shipping_price": shipping_price,
            "time_sold_months": time_sold_months,
            "success_metric": success_val,
            "condition_name": condition_name.lower(),
            "is_new": is_new,
            "is_used": is_used,
            "feedback_score": feedback_score,
            "feedback_percent": feedback_percent,
            "brand": brand_str,
            "avg_review_rating": avg_review_rating
}#again, no start times seem to be available
Available Metrics/Plots:
Market quality = num_sold / total
Market quality by auction and buy now
Condition (new/used) success metric comparison
Seller feedback success comparison (plot and correlate a sellers avg review score and success)
Review rating vs success comparison(plot and correlate)
Brand analysis(size of brand selling and their success)
Success metric vs shipping cost (plot and correlate)
Success for shipped vs not shipped (local pickup) (bar plot)
Auction vs buy now success and final price (bar plot)
Success metric vs number of images comparison (plot and correlate)
Histogram of success metric to show distribution 
Success and price over the last 90 days (plot)

How these metrics help a seller:
Market quality indicates the proportion of items that successfully sold compared to all attempted listings. This helps a seller gauge overall demand and competition in the market: if market quality is high, there’s more chance their item will sell quickly; if it’s low, they may need to adjust pricing or listing format. 
Market quality for auction vs buy_now shows how well items sell between these two selling methods. A seller can determine whether auctions or buy-now listings typically result in higher conversion rates, informing them on which format to choose for their products.
New/used success metric can guide sellers in setting realistic expectations for used products or realizing the premium they might gain on new products.
Seller feedback vs success metrics can help sellers understand how their reputation impacts final sale outcomes.
Product ratings vs success reveals how an item’s price is influenced by its rating. This can help sellers consider how their product quality and review management strategies may affect their income.
Brand analysis tracks how different brands are performing, including the volume of listings and average success. Sellers gain insight into which brands attract more bids or higher final prices, which can inform them towards sale strategies and help them evaluate competition.
Shipping cost vs success allows sellers to evaluate whether free or lower shipping can boost final prices or sales rates, or if higher shipping fees deter buyers.
Shipped vs Not Shipped succes can help sellers evaluate if they can be successful selling locally or if they need to consider expanding their shipping range.
Auction vs Buy-now success and final price plots will allow sellers to make data-based decisions about which format might yield higher profits or success rates for their particular items.
Success vs image # can help a seller evaluate the extent to which photos draw in more buyers 
Histogram of succes showcases the distribution of success values across all items. A seller can quickly see if most items cluster around moderate success or if there’s a substantial tail of very high, or low, performers, indicating potential room for improvement.
Success and price plotted over the last 90 days will help the seller visualize the current market trends and decide if it is a good time to sell or not.

Original (naive) Plan:

Inputs:
Product keywords, intended sale area, # years to analyze

Use Keywords to get listings within intended sale area over past X years:
Market evaluation: a measure of generally how successful similar sellers have been(sellers compared within their year for the past X amount of years) 
Success metric: 
Cluster in two groups: ‘sold successfully and used auction’ and ‘sold successfully and used buy now’. Dont include unsuccessful sells in clustering, but keep count of them for market quality metric.
Plot {final sale price vs time sold} and calculate correlation metric
For every year within each successful cluster rank by sale success metric = (price sold)*min(1.5,(1/(timeSold + 2) + 1))(this asymptotes to 1 and rewards selling within a day by multiplying by 1.5) and calculate avg success metric for that cluster for that year.
Calculate scaled sale success metric = (success metric)*(most recent years avg success metric)/(this years avg success metric)
For each of the auction/buy now groups, cluster all successful listings together with their scaled success metric. Now there should be two clusters total which hold the listing id and scaled success metric for every listing we are observing for each group (auction/ buy now) respectively 
Also calculate market quality metric = (1/#years) * sum(#sold / (#sold + #notsold)) Across all years. 
Features to optimize by success metric: buy now or auction, seller reserve price, post keywords, sale radius, number of pictures, time posted
Plot avg success metric over years for both sale methods
Extract feature values for all listings
Seller reserve price
Keywords that weren’t part of the input
Sale radius
Number of pictures
Time of day posted 
Day of week posted
Week of the year posted
Considering all successful listings across every year for each sale method, plot {scaled success metric vs seller reserve price, scaled success metric vs sale radius, scaled success metric vs number of pictures, scaled success metric vs time of day posted }calculate correlation value for all of these plots
For all listings, bar chart plot {avg scaled success metric vs day of week, avg scaled success metric vs week in year posted}
Extract the top 25% of listings ranked by scaled success metric for both groups: auction/buynow
Calculate other output metrics:
Avg reserve price
Mode keywords (not case sensitive)(~top 5)
Avg sale radius
Avg picture #
Avg time of day posted 
Mode day of week posted
Avg scaled success metric
Avg final sale price
Competition over time and demand over time(lower granularity)
For all listings observed during success metric analysis, also be counting competition metric
For every year keep a table associating weeks out of the year with number of sellers 
Competition metric = For every week take avg of number of sellers across all years on that week
Next interpret demand over time
Again associate each week with the number of queries containing some (~60%) of your keywords that week
Demand metric = Avg this number across the same week for all years
Plot bar chart {avg # of listings this week over all years vs week in the year, avg # of queries this week over all years vs week in year}
Get optimal time of year metric = 
argmax_(week){demand(week) / competition(week))}
Outputs:
All plots and their correlation values as well as all bar charts
Output metrics for auction and for buy now
Optimal time of year metric
Market quality metric
Summary (“If you are trying to sell by auction it is suggested that you use {optimal features(keywords, reserve price, sale radius, picture #, time of day, day of week)} and expect a sale around {avg final sale price} dollars. To optimize for buyer demand against competing sellers aim for the {optimal time of year metric} week in the year. The average scaled success metric is {avg scaled success metric}. If you want to sell through ‘buy now’, it is suggested … {same as before}. Be aware the the market quality metric is {market quality metric} (probability of post successfully selling)”) 

Future Plans:
Implementing more advanced consideration in success metric(not just optimizing for price sold but also maximizing visibility success)
Larger data models for more advanced modeling and prediction

Plots:
Final sale price vs. time sold) to see the relationship between sale price and how long it took to sell
Plot Average Success Metric Over Years: For both sale methods, track how the average success metric changes year by year.
Scaled success metric vs features
Comp and demand over week of year
Useful Data:
Price Trend History(past)(comp analysis):
Final sale price
Seller Seasonal fluctuations
Listing duration (create a sense of urgency raises prices or more time raises prices?)
Clickthrough analysis:
How many clicks did similar listings get and how successful their sale was
Auction end time and final price
Watcher count and peak watcher times
Listing Volume (competition analysis):
Active listings(cur data)
How many listings sold vs unsold and for what price
Competitiveness for this type of item (avg active listings per month over a time span)
Peak buyer times(deman analysis)
Consider for target buyer time zone, peak browsing day of the week and hours
Buyer seasonality
Keyword analysis(demand analysis)
Identify when product keywords are most searched by users
Identify when product keywords are most offered by competing sellers
Suggest keyword synonyms to improve visibiltiy based on doing this same analysis with other keywords and selecting the ones that improve it the most
