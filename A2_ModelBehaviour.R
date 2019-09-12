#Assignment 2
#ModelBehaviour 
#PES1201700921 PES1201700246 PES12017001438

#packages used 
# packages that you will be using
#install.packages("RColorBrewer")
#install.packages("tidyr")
#install.packages("textdata")
#install.packages("textstem")
#install.packages("koRpus")
#install.koRpus.lang("en")
#install.packages("plotly")
#install.packages("rgdal")
#install.packages("data.table")
#install.packages("dplyr")
#install.packages("ggplot2")
#install.packages("raster")
#install.packages("grid")
#install.packages("maps")
library(rgdal)
library(data.table)
library(ggplot2)
library(raster)
library(grid)
library(maps)
library(SnowballC)
library(dplyr)
library(tidytext)
library(tidyr)
library(wordcloud)
library(reshape2)
library(koRpus)
library(textstem)
library(plotly)


#paths for each question(specified by the number following the word Path)
Path1="C:\\Users\\diyas\\Desktop\\SEM5\\Data_Analytics\\Assignment2\\dataset\\fitness_data.csv"
Path2="C:\\Users\\diyas\\Desktop\\SEM5\\Data_Analytics\\Assignment2\\dataset\\subject.csv"
Path4='C:\\Users\\diyas\\Desktop\\SEM5\\Data_Analytics\\Assignment2\\dataset\\Lok Sabha-2014 data.csv'
Path5="C:\\Users\\diyas\\Desktop\\SEM5\\Data_Analytics\\Assignment2\\dataset\\tweets.txt"


#Question-1:Library plotly helps make brilliant visualizations that are highly interactive and appealing. 
#1a.)Use this library to visualize all the parameters in fitness_data.csv(except activityID, subID and timestamp(s)) against timestamp(x-axis) for each subID 
#1b.)To represent in a single graph, use dropdown menus. First dropdown menu will be used to select the subID. Second dropdown menu will be used to select the column to be analyzed. (For eg: User wants to visualize heart rate activity column(dropdown 1) for subID 104(dropdown 2)) 
#1c) Use a slider to control the range of timestamp.
#1d)What insights could you glean from this plot? (There is no single correct answer)

s<-read.csv(file=Path1)
#reading csv and storing into dataframe fitness_data
fitness_data <- data.frame(read.csv(file=Path1,header=TRUE, sep=","))
#fitness_data=na.omit(fitness_data)
x<-dim(fitness_data)
#x
#colnames(fitness_data)

transforms=list(list(type='filter',target=~subID,operation='=',value=unique(fitness_data$subID)))
#transforms
#getting names of all columns and choosing only the attributes that need to be analysed against timestamp
colNames<- names(fitness_data)
#colNames
colNames<-colNames[-which(colNames=='timestamp..in.seconds.')]
colNames<-colNames[-which(colNames=='activityID')]
colNames<-colNames[-which(colNames=='subID')]
#colNames

df<- fitness_data
## Adding trace directly here with first attribute of heart rate
p <- plot_ly(
  
  fitness_data,
  type = "scatter",
  mode = "lines",
  x = ~timestamp..in.seconds.,
  #add_trace(y = ~heartrate.during.activity..bpm., name = 'trace 0',mode = 'lines') 
  y = ~heartrate.during.activity..bpm., 
  name=colNames[1],
  #creating a filter to choose based on "subID"
  transforms=list(list(type='filter',target=~subID,operation='=',value=unique(fitness_data$subID)[1])),
  height=500
)
for (col in colNames[-1]) {
  #adding the rest of the attributes that need to be analyzed against timestamp
  p <- p %>% add_lines(x = ~c, y = fitness_data[[col]], name = col, visible = FALSE)
}
p <- p %>%
  layout(
    title = "Variation of parameters as subject performs activity over time ",
    #creating a slider for the timestamp range(x axis)
    xaxis = list(title = "timestamp(in seconds)",rangeslider = list(type = "int",thickness=0.2)),
    yaxis = list(title = colNames[1]),
    showlegend = FALSE,
    updatemenus = list(
      #dropdown for each unique subID
      list( type='dropdown',active=0,
            x=1,
            y=1.6,
            buttons=list(
              list(method="restyle",args=list("transforms[0].value",unique(fitness_data$subID)[1]),label=unique(fitness_data$subID)[1]),
              list(method="restyle",args=list("transforms[0].value",unique(fitness_data$subID)[2]),label=unique(fitness_data$subID)[2]),
              list(method="restyle",args=list("transforms[0].value",unique(fitness_data$subID)[3]),label=unique(fitness_data$subID)[3]),
              list(method="restyle",args=list("transforms[0].value",unique(fitness_data$subID)[4]),label=unique(fitness_data$subID)[4]),
              list(method="restyle",args=list("transforms[0].value",unique(fitness_data$subID)[5]),label=unique(fitness_data$subID)[5]),
              list(method="restyle",args=list("transforms[0].value",unique(fitness_data$subID)[6]),label=unique(fitness_data$subID)[6]),
              list(method="restyle",args=list("transforms[0].value",unique(fitness_data$subID)[7]),label=unique(fitness_data$subID)[7]),
              list(method="restyle",args=list("transforms[0].value",unique(fitness_data$subID)[8]),label=unique(fitness_data$subID)[8]),
              list(method="restyle",args=list("transforms[0].value",unique(fitness_data$subID)[9]),label=unique(fitness_data$subID)[9])
            )
      ),
      #dropdown for each column
      list(
        y = 1.9,
        x=1,
        type='dropdown',active=0,
        ## Add all buttons at once using lapply
        buttons = lapply(colNames, function(col) {
          list(method = "update",
               args = list(
                 #"y", list(fitness_data[[col]]), 
                 list("visible"= colNames == col),list("yaxis" =list(title = col))),  # put it in a list
               label = col)
          
        })
      )
      
    )
  )
p

#insights gleaned from the plots
cat("The dataset represents information regarding the fitness of 9 subjects tested across different parameters.
The parameters to measure this were heart rate of the subjects under activity, body temperature. Measurements of Accelaration,
Gyroscope and Magnetometer were also taken.
The timestamp has an approximate range of 0-4000s.

Initially, all the subjects have a low heart rate(around 121 bpm). As the time progessess, due to the strain of activity,
it can be clearly inferred from the graphs of all the subjects that their heart rate increases and crosses a threshold of around 1500 bpm.
 
The normal heart rate for all the subjets appears constant as they are not subjected to any activity.

The graphs showing the body temperature of all the subjects was evidence to the fact that activity generated heat in the body,
thereby, showing a gradual increase in the slope of the graph along the time axis.
A gradual decrease can be observed soon after, implying that the activity lasted for a specified amount of time.

The Accelaration measured in m/s^2 across the x,y and z axis shows constant fluctuation of values which in most cases has an extremely 
sharp rises and falls.

Similarly, Gyroscope and Magnetometer values measured across the x,y and z axis shows constant fluctuation of values which in most cases has an extremely 
sharp rises and falls.

Thus,the Accelaration,Gyroscope and Magnetometer values across x,y and z axis showed no clear pattern. 
 ")

#Question 2:What is undersampling and oversampling? Consider the dataset  subject.csv. Is there a case of undersampling or oversampling? If so, mention a technique to remedy the problem. Justify your answer.

cat("Data imbalance usually reflects an unequal distribution of classes within a dataset.

Undersampling balances the dataset by reducing the size of the majority class. This method is used when quantity of data is sufficient.It is the process where you randomly delete some of the observations from the majority class in order to match the numbers with the minority class.

Oversampling balances the dataset by increasing the size of the minority classes.This method is used when quantity of data is insufficient.It is the process of generating synthetic data that tries to randomly generate a sample of the attributes from observations in the minority class.Rather than getting rid of majority class samples, new rare samples are generated by using e.g. repetition, bootstrapping or SMOTE (Synthetic Minority Over-Sampling Technique) 

In the subject.csv file, the majority class for Dominant hand column is \"right\"(8 cases) while the minority class is \"left\"(1 case). The distribution of data appears skewed and causes a data imbalance(as shown in the graph below)
Oversampling would be an appropriate method to restore the balance of the data as the data for the \"left\" cases of the Dominant hand column is insufficient. New data can be generated synthetic minority oversampling technique (SMOTE). SMOTE algorithm creates artificial data based on feature space (rather than data space) similarities from minority samples.")

#graph demonstrating the skewness
#reading the data from csv file at Path2
df= read.csv(Path2, header = TRUE)
#renaming the column od Dominant Hand as a
names(df)[8] <- "a"
#plotting a barplot to show the imbalance of data
counts <- table(df$a)
barplot(counts, main="Distribution of Dominant Hand",
        xlab="Dominant Hand")

cat("From the graph, it can be inferred that the number of \"right\" entries is far more in abundance compared to \"left\"")


# Question 3:There are various techniques for sampling data. Suggest a sampling technique that you think is ideal for the data in fitness_data.csv, and justify your choice.

cat("The most suitable sampling technique that can be used here is 
Simple Random Sampling. Statefied samplimg based on the strata of age would have been more ideal as the data is related to fitness and age happens to be an important factor. However, since information regarding age is not available to us, Simple random sampling would be the most appropriate sampling technique. Selecting subjects completely at random from the larger population also 
yields a sample that is representative of the population itself.")


#Question 4: In August 2018, Election Commission of India made Lok sabha 2014(Lok Sabha-2014 data.csv) data public so that analysts can use it for 2019 Lok Sabha election. Provide a suitable visualisation that accounts for the distribution of votes across the country.

loksabha=read.csv(Path4, header=TRUE)
loksabha<-na.omit(loksabha)
required <- subset(loksabha,select=c("longitude","latitude","VOTERS.TURN.OUT..IN...","Number.of.voters","MARGIN","PARTY","CONSTITUENCY"))
names(required)[names(required) == "VOTERS.TURN.OUT..IN..."] <- "Turnout"
required$Turnout=as.numeric(levels(required$Turnout))[required$Turnout]
required$Turnout<- required$Turnout/10
#{r fig.height = 8}
#Removing the notable outlier whos coordinates do not lie in the region of elections. outlier was removed and not replaced as each constituency has a specific value and the value cannot be written along with the value of the outlier
outvals= boxplot(required$longitude,plot=FALSE)$out
required <-subset(required,!(longitude %in% outvals))
world <- map_data('world') %>% data.table() #getting the world map
world <- world[region=="India",] #getting India's region
#plotting the map of india
g <- ggplot(world, aes(long, lat)) + 
  geom_polygon(aes(group=group),fill="white",colour="black",size=0.1) +
  coord_equal() + 
  scale_x_continuous(expand=c(0,0)) + 
  scale_y_continuous(expand=c(0,0)) +
  labs(x='Longitude', y='Latitude') +
  theme_bw()
#plotting the margin of victory with the party in the regions mentioned
g <- g + 
  geom_point(data=required,aes(longitude, latitude,colour=PARTY,size=Turnout, alpha=MARGIN))+
  #scale_color_gradient(low="blue", high="red")
  #scale_colour_gradient2()
  theme(
    legend.position = 'left',
    legend.key.size = unit(2, "mm")) 
#plot presentation
g

#Question 5:Many good Bollywood movies were released in 2019, one of them being Kabir Singh. The file tweets.txt contains what people have tweeted about this movie. Provide suitable visualization that depicts the generals sentiment of the audience. 

#reading the tweets data from the text file
text <- readLines(Path5)

#making each tweet a vector
text <- c(text)

#lemmatize word in the text string
text=lemmatize_strings(text)

#creation of a tibble
#tibble:modern dataframe containing important features
#12,596 tweets
text_df <- tibble(line = 1:12596, text = text)

#Tokenization: split each row so that there is one token (word) in each row of the new data frame(lowercase by default)
text_df<-text_df %>%
  unnest_tokens(word, text)

#removing stopwords
data(stop_words)
text_df <- text_df%>%
  anti_join(stop_words)

#sorting of the words in the dataframe
#intermediate output of the datframe displayed on screen
text_df%>%
  count(word, sort = TRUE) 


#Stemming of words
text_df<-text_df %>%
  mutate(word_stem = wordStem(word, language="english"))

#plotting the wordcloud
#using the lexicon "bing" 
#each word in the above generated dataframe is associated with a sentiment and classfied as shown in the plot
text_df %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  acast(word ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = c("red", "darkgreen"),
                   max.words = 150)

#Reference used for Question5: https://www.tidytextmining.com/tidytext.html
