"""
Social Media Analytics Project
Name: Khadeer
Roll Number:
"""

import hw6_social_tests as test

project = "Social" # don't edit this

### PART 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]

'''
makeDataFrame(filename)
#3 [Check6-1]
Parameters: str
Returns: dataframe
'''
def makeDataFrame(filename):
    df = pd.read_csv(filename)
    return df


'''
parseName(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseName(fromString):
    start = fromString.find(":") + 1
    end = fromString.find("(")
    pname = fromString[start:end].strip()   # person's name
    return pname


'''
parsePosition(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parsePosition(fromString):
    start = fromString.find("(") + 1
    end = fromString.find("from")
    ppos = fromString[start:end].strip()    # person's position
    return ppos


'''
parseState(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseState(fromString):
    start = fromString.find("from") + 4
    end = fromString.find(")")
    pstate = fromString[start:end].strip()
    return pstate


'''
findHashtags(message)
#5 [Check6-1]
Parameters: str
Returns: list of strs
'''
def findHashtags(message):
    message += " "
    hmessage = []               #Hashtag Message

    for i in range(len(message)):
        if message[i] == "#":
            start = i
            for j in range(i+1,len(message)):
                if message[j] in endChars:
                    end = j
                    hmessage.append(message[start:end])
                    break

    return hmessage


'''
getRegionFromState(stateDf, state)
#6 [Check6-1]
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    dfRegion = stateDf.loc[stateDf["state"] == state, "region"]
    region = dfRegion.values[0]
    return region


'''
addColumns(data, stateDf)
#7 [Check6-1]
Parameters: dataframe ; dataframe
Returns: None
'''
def addColumns(data, stateDf):
    names = [] ; positions = [] ; states = [] ; regions = [] ; hashtags = []
    for index, row in data.iterrows():

        names.append(parseName(row["label"]))
        positions.append(parsePosition(row["label"]))
        state = parseState(row["label"]) ; states.append(state)   # This statement has been used for ease and understanding 
        regions.append(getRegionFromState(stateDf, state))        # regions.append(getRegionFromState(stateDf, parseState(row["label"])))
        hashtags.append(findHashtags(row["text"]))

    data["name"] = names
    data["position"] = positions
    data["state"] = states
    data["region"] = regions
    data["hashtags"] = hashtags

    return


### PART 2 ###

'''
findSentiment(classifier, message)
#1 [Check6-2]
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    score = classifier.polarity_scores(message)['compound']
    # print(score)
    if score < -0.1:
        # print( "negative" )
        return "negative"
    elif score > 0.1:
        # print( "positive" )
        return "positive"
    else:
        # print( "neutral" )
        return "neutral"


'''
addSentimentColumn(data)
#2 [Check6-2]
Parameters: dataframe
Returns: None
'''
def addSentimentColumn(data):
    classifier = SentimentIntensityAnalyzer()
    sentiment = []
    for index, row in data.iterrows():
        sentiment.append(findSentiment(classifier, row["text"]))
    
    # print(sentiment)
    data["sentiment"] = sentiment
    return


'''
getDataCountByState(data, colName, dataToCount)
#3 [Check6-2]
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''
def getDataCountByState(data, colName, dataToCount):
    statecountdict = {}
    # print(data["sentiment"]); # print(data["message"]); # print(data["bias"])
    if colName == "" and dataToCount == "" :
        for index, row in data.iterrows():
            if row["state"] not in statecountdict:
                statecountdict[row["state"]] = 1
            else:
                statecountdict[row["state"]] += 1
    else:
        for index, row in data.iterrows():
            if row[colName] == dataToCount:
                if row["state"] not in statecountdict:
                    statecountdict[row["state"]] = 1
                else:
                    statecountdict[row["state"]] += 1
    # print(statecountdict)
    return statecountdict


'''
getDataForRegion(data, colName)
#4 [Check6-2]
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    regiondict = {}
    for index, row in data.iterrows():
        if row["region"] not in regiondict:
            regiondict[row["region"]] = {}
            regiondict[row["region"]][row[colName]] = 1
        else: 
            if row[colName] not in regiondict[row["region"]]:
              regiondict[row["region"]][row[colName]] = 1        
            else:
                regiondict[row["region"]][row[colName]] += 1
    return regiondict


'''
getHashtagRates(data)
#5 [Check6-2]
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):
    hashtagsdict = {}
    for index, row in data.iterrows():
        if len(row["hashtags"]) != 0 :
            for i in range(len(row["hashtags"])):
                if row["hashtags"][i] not in hashtagsdict:
                    hashtagsdict[row["hashtags"][i]] = 1
                else:
                    hashtagsdict[row["hashtags"][i]] += 1

    return hashtagsdict


'''
mostCommonHashtags(hashtags, count)
#6 [Check6-2]
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def mostCommonHashtags(hashtags, count):
    hashtagssortbyval = {k:v for k, v in sorted(hashtags.items(), key=lambda v:v[1], reverse=True)}
    topcounthashtagsdict = {}
    for i in hashtagssortbyval:
        if len(topcounthashtagsdict) < count:
            topcounthashtagsdict[i] = hashtagssortbyval[i]

    return topcounthashtagsdict


'''
getHashtagSentiment(data, hashtag)
#7 [Check6-2]
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):
    hashtagsdict = getHashtagRates(data)
    # count = hashtagsdict[hashtag]
    # print("count", count)
    count = 0
    indhashtagmssgscore = 0
    for index, row in data.iterrows():
        if findHashtags(hashtag)[0] in row["hashtags"]:
            count += 1
            if row["sentiment"] == "positive":
                indhashtagmssgscore += 1
            elif row["sentiment"] == "negative":
                indhashtagmssgscore -= 1

    sentimentscore = indhashtagmssgscore/count
    # print("count", count)
    return sentimentscore


### PART 3 ###

'''
graphStateCounts(stateCounts, title)
#2 [Hw6]
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    return


'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
#3 [Hw6]
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    return


'''
graphRegionComparison(regionDicts, title)
#4 [Hw6]
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
    return


'''
graphHashtagSentimentByFrequency(data)
#4 [Hw6]
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    return


#### PART 3 PROVIDED CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    import matplotlib.pyplot as plt

    w = 0.8 / len(labelList)  # the width of the bars
    xPositions = []
    for dataset in range(len(labelList)):
        xValues = []
        for i in range(len(xLabels)):
            xValues.append(i - 0.4 + w * (dataset + 0.5))
        xPositions.append(xValues)

    for index in range(len(valueLists)):
        plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

    plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.ylim(-1, 1)

    # a bit of advanced code to draw a line on y=0
    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    # test.testParseName()
    # test.testParsePosition()
    # test.testParseState()
    # test.testFindHashtags()
    # test.testGetRegionFromState()
    # test.testAddColumns()
    # print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    # test.week1Tests()
    # print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    # test.runWeek1()

    ## Uncomment these for Week 2 ##
    # test.testFindSentiment()
    # test.testAddSentimentColumn()
    # df = makeDataFrame("data/politicaldata.csv")
    # stateDf = makeDataFrame("data/statemappings.csv")
    # addSentimentColumn(df)
    # addColumns(df, stateDf)
    # test.testGetDataCountByState(df)
    # test.testGetDataForRegion(df)
    # test.testGetHashtagRates(df)
    # test.testMostCommonHashtags(df)
    # test.testGetHashtagSentiment(df)
    # print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    # test.week2Tests()
    # print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    # test.runWeek2()

    ## Uncomment these for Week 3 ##
    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()
