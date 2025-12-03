import pyreadr

result = pyreadr.read_r("data/raw/StanfordOpenPolicing/yg821jf8611_ct_statewide_2020_04_01.rds")

df = result[None]

print(df.head())
print(df.sample(1))
print(df.columns)