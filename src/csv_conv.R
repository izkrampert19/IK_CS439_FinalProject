# SIMPLE CODE TO CONVERT THE ORIGINAL .rds FILES TO .csv

# file.exists("../data/raw/StanfordOpenPolicing/yg821jf8611_ct_statewide_2020_04_01.rds")

 df <- readRDS("../data/raw/StanfordOpenPolicing/yg821jf8611_ct_statewide_2020_04_01.rds")
 write.csv(df, "../data/raw/StanfordOpenPolicing/ct_2020_04_01.csv", row.names = FALSE)

# To ensure that data has been copied:
 head(df)