import os
import wget
from datetime import datetime
import itertools
import pandas as pd
import numpy as np
import math
import jsonlines, json, csv
import sys
sys.path.append("../")
import tqdm
from twarc import Twarc
import glob

# These keys are received after applying for a twitter developer account
import jsonlines, json, csv
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""
t = Twarc(consumer_key, consumer_secret, access_token, access_token_secret)

## Paths

data_url = "https://raw.githubusercontent.com/lopezbec/COVID19_Tweets_Dataset/master/Summary_Details/"
output_dir = "Hydrated_tweets/"
tweet_ID_dir = "Tweet_IDs/"
tweet_summary_dir = "Tweet_Summary/"
# create a folder to store tweet IDs if not exists
os.makedirs(tweet_summary_dir, exist_ok=True)
os.makedirs(tweet_ID_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

## define months to study
data_month_dict = {   
    "202201": {
        "start_date": "2022-1-01",
        "end_date": "2022-1-31"},   
}

data_hours = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']

print("downloading ids...")


for data_month, date_range in data_month_dict.items():
    start_date = date_range["start_date"]
    end_date = date_range["end_date"]
    
    dates_list = pd.date_range(start_date, end_date).tolist()
    month_str = dates_list[0].strftime("%Y_%m")
    dates_list = [d.strftime("%Y_%m_%d") for d in dates_list]

    files_list = [
        f"{data_url}{month_str}/{date_str}_{hour_str}_Summary_Details.csv"
        for date_str, hour_str
        in itertools.product(dates_list, data_hours)
    ]
    
    month_directory = f"{tweet_summary_dir}{data_month}"
    os.makedirs(month_directory, exist_ok=True)
    for file in files_list:
        if not os.path.exists(file):
            try:
                wget.download(file, out=month_directory)
            except:
                print("someting went wrong")
                # there are some known gaps with no data collected:
                # https://github.com/lopezbec/COVID19_Tweets_Dataset#data-collection-process-inconsistencies
                pass
            
print("\n\n\n\n\n\npreselecting tweets...")

# create a folder to store tweet IDs if not exists
os.makedirs(tweet_ID_dir, exist_ok=True)

for data_month, date_range in data_month_dict.items():
    print(data_month)
    files = glob.glob(f"{tweet_summary_dir}{data_month}/*.csv")
    tweet_ids = []
    for file in tqdm.tqdm(files):
        data = pd.read_csv(file)

        # only keep English tweets
        data = data[data['Language']=='en']
        # filter out retweets
        data = data[data["RT"]=="NO"] 
        tweet_ids.extend(data["Tweet_ID"])

    # write Tweet IDs to a file for hydration later
    tweet_ids_filename = f"{tweet_ID_dir}/{data_month}.txt"
    with open(tweet_ids_filename, "w+") as f:
        for tweet_id in tweet_ids:
            f.write(f"{tweet_id}\n")

n_split = 10  ## split the data to make the files smaller. 

print("\n\n\n\n\n\nwrite tweet ids...")
# iterate through tweet IDs for each month and sample 10%
for data_month in data_month_dict.keys():
    filename = f"{tweet_ID_dir}{data_month}.txt"
    print(filename)
    # read monthly tweet IDs
    tweet_ids = pd.read_csv(filename, header=None, dtype=str)
    
    # split the data frame into n chunks
    end_i = tweet_ids.shape[0]
    chunk_size = math.ceil(end_i / n_split)

    # iterate through all the chunks and output to file
    for i, start_i in enumerate(range(0, end_i, chunk_size)):
        tweet_split_i = tweet_ids[start_i:start_i + chunk_size]

        # output to a file for each split
        tweet_sample_ids_filename = f"{tweet_ID_dir}{data_month}_{i}.txt"
        with open(tweet_sample_ids_filename, "w+") as f:
            for tweet_id in tweet_split_i[0]:
                f.write(f"{tweet_id}\n")

print("\n\n\n\n\n\nhydrating...")
for data_month in data_month_dict.keys():
    os.makedirs(f"{output_dir}-{data_month}", exist_ok=True)

    for i in range(0, n_split):
        tweet_ids_filename = f"Tweet_IDs/{data_month}_{i}.txt" #@param {type: "string"}
        output_filename = f"{output_dir}-{data_month}/{data_month}_{i}.txt" #@param {type: "string"}
        print("On file %s"%output_filename)
        ids = []
        with open(tweet_ids_filename, "r") as ids_file:
            ids = ids_file.read().split()
        hydrated_tweets = []
        ids_to_hydrate = set(ids)
        # Check hydrated tweets
        if os.path.isfile(output_filename):
            with jsonlines.open(output_filename, "r") as reader:
                for i in reader.iter(type=dict, skip_invalid=True):
                    hydrated_tweets.append(i)
                    ids_to_hydrate.remove(i["id_str"])
        if ids_to_hydrate == 0:
            print("Finished downloading. Skipping.")
            continue

        print("Total IDs: " + str(len(ids)) + ", IDs to hydrate: " + str(len(ids_to_hydrate)))
        print("Hydrated: " + str(len(hydrated_tweets)))
        
        pbar = tqdm.tqdm(total=len(ids_to_hydrate))
        count = len(hydrated_tweets)
        start_index = count

        num_save  = 10000

        # start hydrating
        for tweet in t.hydrate(ids_to_hydrate):
            hydrated_tweets.append(tweet)
            count += 1
            # If num_save iterations have passed,
            if (count % num_save) == 0:
                with jsonlines.open(output_filename, "a") as writer:
                    for hydrated_tweet in hydrated_tweets[start_index:]:
                        writer.write(hydrated_tweet)
                start_index = count
            pbar.update(1)

        if count != start_index:
            print("Here with start_index", start_index)
            with jsonlines.open(output_filename, "a") as writer:
                for hydrated_tweet in hydrated_tweets[start_index:]:
                    writer.write(hydrated_tweet)  
