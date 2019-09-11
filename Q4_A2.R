install.packages("rgdal")
install.packages("data.table")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("raster")
install.packages("grid")
install.packages("maps")
library(rgdal)
library(data.table)
library(dplyr)
library(ggplot2)
library(raster)
library(grid)
library(maps)
#Creating a dataframe of required data
path1='C:\\Users\\chira\\OneDrive\\Desktop\\Semester 5\\Data Analytics\\Assignment 2\\DA19_A2_Datasets\\dataset\\Lok Sabha-2014 data.csv'
loksabha=read.csv(path1, header=TRUE)
loksabha<-na.omit(loksabha)
required <- subset(loksabha,select=c("longitude","latitude","VOTERS.TURN.OUT..IN..."))
names(required)[names(required) == "VOTERS.TURN.OUT..IN..."] <- "Turnout"
required$Turnout=as.numeric(levels(required$Turnout))[required$Turnout]


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
#plotting the % of voter turn out in the regions mentioned
g <- g + 
  #geom_tile(data=required,aes(longitude, latitude, fill=Turnout)) +
stat_summary2d(data=required,aes(longitude, latitude, z=Turnout))+
  scale_fill_gradient(low="blue", high="red", name='Voter Turn Out %') +
  theme(
    legend.position = 'bottom',
    legend.key.size = unit(1, "cm")
  ) 
#plot presentation
g
