#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3


# #__Database Management__
# 
# ##__*Question 1*__

# In[2]:


# Creates SQlite database
carsharing = sqlite3.connect("carsharing.db", isolation_level = None)
cur = carsharing.cursor()
print("Database created successfully")


# Result:
# Database created successfully

# In[3]:


# Creates CarSharing table
create_table_query = """
CREATE TABLE CarSharing
(ID INTEGER PRIMARY KEY NOT NULL,
TIMESTAMP DATETIME,
SEASON TEXT,
HOLIDAY TEXT,
WORKINGDAY TEXT,
WEATHER TEXT,
TEMP REAL,
TEMP_FEEL REAL,
HUMIDITY REAL,
WINDSPEED REAL,
DEMAND REAL); """

cur.execute(create_table_query)
print("Table created successfully")


# Result:
# Table created successfully

# In[4]:


#Opens the CSV file and inserts its contents into CarSharing table

with open("CarSharing.csv", "r") as csvfile:
    next(csvfile)
    for line in csvfile:
        row = line.strip().split(",")
        cur.execute("""INSERT INTO CarSharing (ID, TIMESTAMP, SEASON, HOLIDAY, WORKINGDAY, WEATHER, TEMP, TEMP_FEEL, HUMIDITY, WINDSPEED, DEMAND)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""", tuple(row));
        
print("Data inserted into CarSharing table successfully.")


# Result:
# Data inserted into CarSharing table successfully.

# In[5]:


# Creates a backup table named "CarSharing_Backup" and copy data

CarSharing_backup_table = """
CREATE TABLE CarSharing_Backup AS
SELECT * FROM CarSharing;
"""

cur.execute(CarSharing_backup_table)

print("Back up table created and data copied successfully")


# Result:
# Back up table created and data copied successfully

# ##__*Question 2*__

# In[6]:


# add "temp_category" column to CarSharing table 

cur.execute("""ALTER TABLE CarSharing
ADD COLUMN TEMP_CATEGORY TEXT;
""")


# Result:
# <sqlite3.Cursor at 0x156067eb730>    

# In[7]:


# Update temp_category column based on feels-like temperature

update_temp_category_column = """
UPDATE CarSharing
SET TEMP_CATEGORY = 
    CASE 
        WHEN TEMP_FEEL < 10 THEN 'Cold'
        WHEN TEMP_FEEL >= 10 AND TEMP_FEEL <= 25 THEN 'Mild'
        WHEN TEMP_FEEL >25 THEN 'Hot'
        ELSE " "
    END;
"""
cur.execute(update_temp_category_column)
print("TEMP_CATEGORY column added and updated successfully.")


# Result:
# TEMP_CATEGORY column added and updated successfully.

# In[ ]:





# ##__*Question 3*__

# In[8]:


# Create the temperature table by selecting temp,temp_feel =, and temp_category columns from CarSharing
cur.execute('''
CREATE TABLE Temperature AS
SELECT ID, TEMP, TEMP_FEEL, TEMP_CATEGORY
FROM CarSharing
''')
print("Temperature table created successfully")


# Result:
# Temperature table created successfully

# In[9]:


# Create a new version of the CarSharing table without temp and feels_like columns
cur.execute('''
CREATE TABLE CarSharing_new AS
SELECT ID, TIMESTAMP, SEASON, HOLIDAY, WORKINGDAY, WEATHER, HUMIDITY, WINDSPEED, DEMAND, TEMP_CATEGORY
FROM CarSharing
''')
print("New version of CarSharing table created successfully!")


# Result:
# New version of CarSharing table created successfully!

# In[10]:


# Drop the original CarSharing table
cur.execute('DROP TABLE CarSharing')


# In[11]:


# Rename the new table to CarSharing
cur.execute('ALTER TABLE CarSharing_new RENAME TO CarSharing')


# Result:
# <sqlite3.Cursor at 0x156067eb730>

# ##__*Question 4*__

# In[12]:


# Find distinct weather conditions
cur.execute('SELECT DISTINCT weather FROM CarSharing')
distinct_weather = cur.fetchall()
print(distinct_weather)


# Result:
# 
# [('Clear or partly cloudy',), ('Mist',), ('Light snow or rain',), ('heavy rain/ice pellets/snow + fog',)]

# In[13]:


# Assign a unique number to each distinct weather condition
weather_codes = {weather[0]: index for index, weather in enumerate(distinct_weather)}

print(weather_codes)


# Result:
# {'Clear or partly cloudy': 0, 'Mist': 1, 'Light snow or rain': 2, 'heavy rain/ice pellets/snow + fog': 3}

# In[14]:


# Add the weather_code column
cur.execute('ALTER TABLE CarSharing ADD COLUMN WEATHER_CODE INTEGER')

print("Weather_code column add to CarSharing table successfully!")


# Result:
# Weather_code column add to CarSharing table successfully!

# In[15]:


# Update weather_code column for each weather condition
for weather, code in weather_codes.items():
    cur.execute('''
    UPDATE CarSharing
    SET WEATHER_CODE = ?
    WHERE weather = ?
    ''', (code, weather))

print("Weather_code column updated successfully!")


# Results:
# Weather_code column updated successfully!

# In[ ]:





# ##__*Question 5*__

# In[16]:


# Create the weather table
cur.execute('''
CREATE TABLE Weather AS
SELECT ID, WEATHER, WEATHER_CODE
FROM CarSharing
''')

print("Weather table created successfully!")


# Result:
# Weather table created successfully!

# In[17]:


# Create a new version of CarSharing without the weather column
cur.execute('''
CREATE TABLE CarSharing_new AS
SELECT ID, TIMESTAMP, SEASON, HOLIDAY, WORKINGDAY, HUMIDITY, WINDSPEED, DEMAND, TEMP_CATEGORY, WEATHER_CODE
FROM CarSharing
''')

print("New version of CarSharing table created successfully!")


# Result:
# New version of CarSharing table created successfully!

# In[18]:


# Drop the original CarSharing table
cur.execute('DROP TABLE CarSharing')


# Result:
# <sqlite3.Cursor at 0x156067eb730>

# In[19]:


# Rename the new table back to CarSharing
cur.execute('ALTER TABLE CarSharing_new RENAME TO CarSharing')


# Result:
# <sqlite3.Cursor at 0x156067eb730>

# In[ ]:





# ##__*Question 6*__

# In[20]:


# Create the "time" table
cur.execute('''
CREATE TABLE Time (
    ID INTEGER PRIMARY KEY,
    TIMESTAMP DATETIME,
    HOUR INTEGER,
    WEEKDAY TEXT,
    MONTH TEXT
)
''')
print(" The time table created successfully!")


# Result:
# The time table created successfully!

# In[21]:


# Insert data into the time table
cur.execute('''
INSERT INTO Time (ID,TIMESTAMP, HOUR, WEEKDAY, MONTH)
SELECT
    ID,
    TIMESTAMP,
    CAST(strftime('%H', TIMESTAMP) AS INTEGER) AS hour,
    CASE strftime('%w', TIMESTAMP) 
        WHEN '0' THEN 'Sunday'
        WHEN '1' THEN 'Monday'
        WHEN '2' THEN 'Tuesday'
        WHEN '3' THEN 'Wednesday'
        WHEN '4' THEN 'Thursday'
        WHEN '5' THEN 'Friday'
        WHEN '6' THEN 'Saturday'
    END AS Weekday,
    CASE strftime("%m", TIMESTAMP)
        WHEN "01" THEN "January"
        WHEN "02" THEN "February"
        WHEN "03" THEN "March"
        WHEN "04" THEN "April"
        WHEN "05" THEN "May"
        WHEN "06" THEN "June"
        WHEN "07" THEN "July"
        WHEN "08" THEN "August"
        WHEN "09" THEN "September"
        WHEN "10" THEN "October"
        WHEN "11" THEN "November"
        WHEN "12" THEN "December"
    END AS Month
FROM CarSharing
''')

print("Data insterted into time table successfully!")


# Result:
# Data insterted into time table successfully!

# ##__*Question 7*__

# a._Which date and time had the highest demand rate in 2017._

# In[22]:


cur.execute("""SELECT TIMESTAMP, MAX(demand) AS max_demand
FROM CarSharing
WHERE TIMESTAMP LIKE '2017-%'
GROUP BY TIMESTAMP
ORDER BY max_demand DESC
LIMIT 1;""")

high_demand_date = cur.fetchall()
print(high_demand_date)


# Result:
# [('2017-06-15 17:00:00', 6.45833828334479)]

# b._A table containing the name of the weekday, month, and season with the highest and lowest average demand rates throughout 2017 as well as the average demand values_

# In[23]:


cur.execute("""SELECT Season, Weekday, Month, avg_Demand, Demand_type FROM (
    SELECT c.Season Season, t.Weekday Weekday, t.Month Month, AVG(c.Demand) avg_Demand, 'Highest' Demand_type
    FROM CarSharing c
    JOIN Time t ON c.Id = t.Id
    WHERE c.Timestamp LIKE '2017-%'
    GROUP BY c.Season, t.Weekday, t.Month
    ORDER BY avg_Demand DESC
    LIMIT 1
) AS HighestDemand
UNION ALL
SELECT Season, Weekday, Month, avg_Demand, Demand_type FROM (
    SELECT c.Season Season, t.Weekday Weekday, t.Month Month, AVG(c.Demand) avg_Demand, 'Lowest' Demand_type
    FROM CarSharing c
    JOIN Time t ON c.Id = t.Id
    WHERE c.Timestamp LIKE '2017-%'
    GROUP BY c.Season, t.Weekday, t.Month
    ORDER BY avg_Demand ASC
    LIMIT 1
) AS LowestDemand
ORDER BY avg_Demand;""") 

highest_lowest_avg_demand = cur.fetchall()

print(highest_lowest_avg_demand)




# Result:
# [('spring', 'Monday', 'January', 3.0507857781010803, 'Lowest'), ('fall', 'Sunday', 'July', 4.997135078747038, 'Highest')]

# In[ ]:





# c._For the weekday selected in (b), a table showing the average demand rate we had at different hours of that weekday throughout 2017. Please sort the results in descending order based on the average demand rates._

# In[24]:


# average demand rates at different hours on Mondays in 2017
cur.execute("""SELECT t.hour Hour, AVG(c.demand) AS avg_demand
FROM CarSharing c
JOIN Time t
ON t.Id = c.Id
WHERE t.weekday = "Monday" AND c.Timestamp LIKE '2017-%'
GROUP BY Hour
ORDER BY avg_demand DESC;""")

print(cur.fetchall())


# Result:
# [(13, 5.643553885574589), (12, 5.621972407564752), (14, 5.5546127640508125), (15, 5.515114816673652), (16, 5.5037531060819065), (11, 5.437364716990808), (17, 5.399252221390464), (10, 5.223831412212205), (18, 5.215942911367383), (19, 4.990499880317549), (20, 4.72698612096739), (9, 4.638344516787047), (21, 4.464855592800993), (0, 4.230481522191123), (22, 4.188674635244383), (1, 3.976928867430087), (8, 3.9349436288968973), (23, 3.7996223936508624), (2, 3.768968806542776), (3, 3.074058279169495), (7, 3.007592707236988), (6, 2.002182356895662), (5, 1.7434286054690797), (4, 1.6598884884502538)]

# In[ ]:





# In[25]:


# average demand rates at different hours on Sundays in 2017
cur.execute("""SELECT t.hour Hour, AVG(c.demand) AS avg_demand
FROM CarSharing c
JOIN Time t
ON t.Id = c.Id
WHERE t.weekday = "Sunday" AND c.Timestamp LIKE '2017-%'
GROUP BY Hour
ORDER BY avg_demand DESC;""")

print(cur.fetchall())


# Result:
# [(15, 5.537925196766501), (14, 5.513702656176965), (16, 5.496274498482708), (13, 5.478758230367251), (12, 5.457459126923962), (17, 5.367634349646671), (11, 5.290915067582866), (18, 5.241211627173081), (10, 5.074545768738012), (19, 5.041802430032878), (20, 4.790666498034493), (9, 4.683176378243488), (21, 4.6042216294700635), (22, 4.47761997279496), (23, 4.3467540044988375), (8, 4.204916044052769), (0, 4.1349741021373445), (1, 3.8698532861655472), (2, 3.6112321135368823), (7, 3.2940243626024213), (3, 2.7702626411657163), (6, 2.453218590596452), (4, 1.6592273354933094), (5, 1.6491762147587958)]

# In[ ]:





# d.i)_What the weather was like in 2017. Was it mostly cold, mild, or hot? 

# In[26]:


cur.execute("""SELECT Temp_Category, COUNT(Temp_category) Prevalence
FROM CarSharing
WHERE TimeStamp LIKE "2017-%"
GROUP BY Temp_Category
ORDER BY Prevalence DESC
LIMIT 1;""")

print(f"The weather was mostly: {cur.fetchall()} in 2017.")


# Result:
# The weather was mostly: [('Mild', 2660)] in 2017.

# In[ ]:





# ii)_Which weather condition (shown in the weather column) was the most prevalent in 2017?_

# In[27]:


cur.execute("""SELECT Weather, COUNT(Weather) Prevalence
FROM CarSharing_Backup
WHERE TimeStamp LIKE "2017-%"
GROUP BY Weather
ORDER BY Prevalence DESC
LIMIT 1;""")

print(f"The weather condition in 2017 was mostly: {cur.fetchall()}")


# Result:
# The weather condition in 2017 was mostly: [('Clear or partly cloudy', 3583)]

# In[ ]:





# iii)_What was the average, highest, and lowest wind speed and
# humidity for each month in 2017? Organise in two tablesfor the wind speed and humidity

# In[28]:


#the average, highest, and lowest wind speed for each month in 2017

cur.execute("""SELECT 
    t.month Month,
    AVG(c.windspeed) AS average_wind_speed,
    MAX(c.windspeed) AS highest_wind_speed,
    MIN(c.windspeed) AS lowest_wind_speed
FROM CarSharing c
JOIN Time t
ON c.Id = t.Id
WHERE c.timestamp LIKE '2017-%'
GROUP BY Month;""")

print("The average, highest, and lowest wind speed for each month in 2017:\n")
print(cur.fetchall())


# Result:
# 
# The average, highest, and lowest wind speed for each month in 2017:
# 
# [('April', 15.538713626373589, '', 0.0), ('August', 12.057296491228056, '', 0.0), ('December', 10.575053728070179, '', 0.0), ('February', 15.123657623318374, '', 0.0), ('January', 13.524766125290023, '', 0.0), ('July', 11.831391885964894, '', 0.0), ('June', 11.568240570175439, '', 0.0), ('March', 15.545066591928244, '', 0.0), ('May', 12.182113377192968, '', 0.0), ('November', 11.822737719298253, '', 0.0), ('October', 10.604789450549458, '', 0.0), ('September', 11.410913465783644, '', 0.0)]

# In[ ]:





# In[29]:


#the average, highest, and lowest humidity for each month in 2017

cur.execute("""SELECT 
    t.month Month,
    AVG(c.humidity) AS average_humidity,
    MAX(c.humidity) AS highest_humidity,
    MIN(c.humidity) AS lowest_humidity
FROM CarSharing c
JOIN Time t
ON c.Id = t.Id
WHERE c.timestamp LIKE '2017-%'
GROUP BY Month;""")

print("The average, highest, and lowest humidity for each month in 2017:\n")
print(cur.fetchall())


# Result:
# The average, highest, and lowest humidity for each month in 2017:
# 
# [('April', 66.1032967032967, '', 22.0), ('August', 62.03728070175438, '', 25.0), ('December', 64.89473684210526, '', 26.0), ('February', 53.58071748878924, 100.0, 8.0), ('January', 56.046403712296986, '', 28.0), ('July', 59.76315789473684, '', 17.0), ('June', 57.98684210526316, '', 20.0), ('March', 55.87219730941704, '', 0.0), ('May', 71.21491228070175, '', 24.0), ('November', 64.02850877192982, '', 27.0), ('October', 71.57142857142857, 100.0, 29.0), ('September', 74.50993377483444, '', 42.0)]  

# In[ ]:





# iv)_Table showing the average demand rate for each cold, mild, and hot weather in 2017 sorted in
# descending order based on their average demand rates. 

# In[30]:


cur.execute("""SELECT Temp_category, AVG(Demand) avg_Demand_rate
FROM CarSharing
WHERE TIMESTAMP LIKE "2017-%"
GROUP BY Temp_category
ORDER BY avg_Demand_rate DESC;""")

print("The average demand rate for each cold, mild, and hot weather in 2017 \n")
print(cur.fetchall())


# Result:
# 
# The average demand rate for each cold, mild, and hot weather in 2017 
# 
# [('Hot', 4.774352585192317), ('Mild', 4.021015429126216), ('Cold', 3.1902527494822346)]

# In[ ]:





# In[ ]:





# e._Table showing the information in (d) for the month with the highest average demand rate in 2017._

# In[31]:


cur.execute("""SELECT t.Month Month, 
temp.Temp_category Temp_category, 
c.Weather Weather, 
AVG(c.Demand) avg_Demand, 
AVG(c.humidity) average_humidity,
MAX(c.humidity) highest_humidity, 
MIN(c.humidity) lowest_humidity,
AVG(c.windspeed) average_wind_speed,
MAX(c.windspeed) highest_wind_speed, 
MIN(c.windspeed) lowest_wind_speed
    
FROM CarSharing_Backup c
JOIN Temperature temp
ON c.Id = temp.Id
JOIN Time t
ON t.Id = temp.Id
WHERE c.Timestamp LIKE "2017-%"
GROUP BY Month, Temp_category, Weather
ORDER BY avg_Demand DESC
LIMIT 1;""")

print("The month with the highest average demand rate in 2017\n")
print(cur.fetchall())
    


# Result:
# The month with the highest average demand rate in 2017
# 
# [('May', 'Hot', 'Clear or partly cloudy', 5.475893095922396, 58.654411764705884, 94.0, 24.0, 13.242747058823536, '', 0.0)]

# In[ ]:





# In[ ]:




