from pyspark.sql import functions
from pyspark.ml.feature import Word2Vec, StopWordsRemover
import itertools


# Loading Data
sqlContext = SQLContext(sc)
df = sqlContext.read.load("data/train.tsv", format="com.databricks.spark.csv", header="true", inferSchema="true", delimiter='\t')


# Viewing
# mean price by item_condition
#df.groupBy("item_condition_id").agg({"price": "mean"}).withColumnRenamed("avg(price)","mean_price").orderBy("mean_price").collect()
#
#
## Get all unique brand names
#join_text = functions.udf(lambda x: " ".join(set(x)))
#df.groupBy("item_condition_id").agg(functions.collect_list("brand_name").alias("brand_name")).withColumn("brand_name", join_text("brand_name")).collect()
#
## Get number of unique brands per item_condition_id
#df.groupBy("item_condition_id").agg(functions.countDistinct("brand_name")).orderBy("item_condition_id").collect()
#
## no of unique brands total
#df.select("brand_name").distinct().count()


def getAnalogy(s, model):
    qry = model.transform(s[0]) - model.transform(s[1]) - model.transform(s[2])
    res = model.findSynonyms((-1)*qry,5) # return 5 "synonyms"
    res = [x[0] for x in res]
    for k in range(0,3):
        if s[k] in res:
            res.remove(s[k])
    return res[0]



df = df.fillna({"item_description": ""})
df = df.withColumn("description_words", functions.split("item_description", " "))

#TODO: Create function that cleans description
def clean_description(text):



word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="description_words", outputCol="word2vec_results")
model = word2Vec.fit(df.select("description_words"))
result = model.transform(df.select("description_words"))
