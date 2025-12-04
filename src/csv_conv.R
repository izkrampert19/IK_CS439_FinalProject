# SIMPLE CODE TO CONVERT THE ORIGINAL .rds FILES TO .csv

# file.exists("../data/raw/StanfordOpenPolicing/yg821jf8611_ct_statewide_2020_04_01.rds")

 df <- readRDS("../data/raw/StanfordOpenPolicing/yg821jf8611_ct_statewide_2020_04_01.rds")
 write.csv(df, "../data/raw/StanfordOpenPolicing/ct_2020_04_01.csv", row.names = FALSE)

# To ensure that data has been copied:
 head(df)

 ## NOTE: DO NOT PUSH CSV FILES REGULARLY! They are too large. You'll need to push them with 
 # Git LFS ... but do this only when all files hve been converted & you make a copy of the csvs
 # in another folder.

 #This comment is to test commit