# (product_id, change_date) is the primary key (combination of columns with unique values) of this table.
# In MySQL, they say the window function performs an aggregate-like operation on a set of query rows. 
# Even though they work almost the same, the aggregate function returns a single row for each target field, 
# but the window function produces a result for each row.
# This table may contain duplicates (In other words, there is no primary key for this table in SQL).
# As all of your values are null, count(cola) has to return zero.
# If you want to count the rows that are null, you need count(*)


# 580. Count Student Number in Departments
# RegEx provides various functionality, here are a few relevant ones:
# ^: This represents the start of a string or line.
# [a-z]: This represents a character range, matching any character from a to z.
# [0-9]: This represents a character range, matching any character from 0 to 9.
# [a-zA-Z]: This variant matches any character from a to z or A to Z. Note that there is no limit to the number of character ranges you can specify inside the square brackets -- you can add additional characters or ranges you want to match.
# [^a-z]: This variant matches any character that is not from a to z. Note that the ^ character is used to negate the character range, which means it has a different meaning inside the square brackets than outside where it means the start.
# [a-z]*: This represents a character range, matching any character from a to z zero or more times.
# [a-z]+: This represents a character range, matching any character from a to z one or more times.
# .: This matches exactly one of any character.
# \.: This represents a period character. Note that the backslash is used to escape the period character, as the period character has a special meaning in regular expressions. Also note that in many languages, you need to escape the backslash itself, so you need to use \\..
# The dollar sign: This represents the end of a string or line.

# 1341. Movie Rating
# Table: Movies
# +---------------+---------+
# | Column Name   | Type    |
# +---------------+---------+
# | movie_id      | int     |
# | title         | varchar |
# +---------------+---------+
# movie_id is the primary key (column with unique values) for this table.
# title is the name of the movie.
 
# Table: Users
# +---------------+---------+
# | Column Name   | Type    |
# +---------------+---------+
# | user_id       | int     |
# | name          | varchar |
# +---------------+---------+
# user_id is the primary key (column with unique values) for this table.
# The column 'name' has unique values.
# Table: MovieRating
# +---------------+---------+
# | Column Name   | Type    |
# +---------------+---------+
# | movie_id      | int     |
# | user_id       | int     |
# | rating        | int     |
# | created_at    | date    |
# +---------------+---------+
# (movie_id, user_id) is the primary key (column with unique values) for this table.
# This table contains the rating of a movie by a user in their review.
# created_at is the user's review date. 
 
# Write a solution to:
# Find the name of the user who has rated the greatest number of movies. In case of a tie, return the lexicographically smaller user name.
# Find the movie name with the highest average rating in February 2020. In case of a tie, return the lexicographically smaller movie name.
# The result format is in the following example.
# Example 1:
# Input: 
# Movies table:
# +-------------+--------------+
# | movie_id    |  title       |
# +-------------+--------------+
# | 1           | Avengers     |
# | 2           | Frozen 2     |
# | 3           | Joker        |
# +-------------+--------------+
# Users table:
# +-------------+--------------+
# | user_id     |  name        |
# +-------------+--------------+
# | 1           | Daniel       |
# | 2           | Monica       |
# | 3           | Maria        |
# | 4           | James        |
# +-------------+--------------+
# MovieRating table:
# +-------------+--------------+--------------+-------------+
# | movie_id    | user_id      | rating       | created_at  |
# +-------------+--------------+--------------+-------------+
# | 1           | 1            | 3            | 2020-01-12  |
# | 1           | 2            | 4            | 2020-02-11  |
# | 1           | 3            | 2            | 2020-02-12  |
# | 1           | 4            | 1            | 2020-01-01  |
# | 2           | 1            | 5            | 2020-02-17  | 
# | 2           | 2            | 2            | 2020-02-01  | 
# | 2           | 3            | 2            | 2020-03-01  |
# | 3           | 1            | 3            | 2020-02-22  | 
# | 3           | 2            | 4            | 2020-02-25  | 
# +-------------+--------------+--------------+-------------+
# Output: 
# +--------------+
# | results      |
# +--------------+
# | Daniel       |
# | Frozen 2     |
# +--------------+
# Explanation: 
# Daniel and Monica have rated 3 movies ("Avengers", "Frozen 2" and "Joker") but Daniel is smaller lexicographically.
# Frozen 2 and Joker have a rating average of 3.5 in February but Frozen 2 is smaller lexicographically.
(SELECT name AS results
FROM MovieRating JOIN Users USING(user_id)
GROUP BY name
ORDER BY COUNT(*) DESC, name
LIMIT 1)

UNION ALL

(SELECT title AS results
FROM MovieRating JOIN Movies USING(movie_id)
WHERE EXTRACT(YEAR_MONTH FROM created_at) = 202002
GROUP BY title
ORDER BY AVG(rating) DESC, title
LIMIT 1)


# 1164. Product Price at a Given Date
# Table: Products
# +---------------+---------+
# | Column Name   | Type    |
# +---------------+---------+
# | product_id    | int     |
# | new_price     | int     |
# | change_date   | date    |
# +---------------+---------+
# (product_id, change_date) is the primary key (combination of columns with unique values) of this table.
# Each row of this table indicates that the price of some product was changed to a new price at some date.
 
# Write a solution to find the prices of all products on 2019-08-16. Assume the price of all products before any change is 10.
# Return the result table in any order.
# The result format is in the following example.
# Example 1:
# Input: 
# Products table:
# +------------+-----------+-------------+
# | product_id | new_price | change_date |
# +------------+-----------+-------------+
# | 1          | 20        | 2019-08-14  |
# | 2          | 50        | 2019-08-14  |
# | 1          | 30        | 2019-08-15  |
# | 1          | 35        | 2019-08-16  |
# | 2          | 65        | 2019-08-17  |
# | 3          | 20        | 2019-08-18  |
# +------------+-----------+-------------+
# Output: 
# +------------+-------+
# | product_id | price |
# +------------+-------+
# | 2          | 50    |
# | 1          | 35    |
# | 3          | 10    |
# +------------+-------+
SELECT
  product_id,
  IFNULL (price, 10) AS price
FROM
  (
    SELECT DISTINCT
      product_id
    FROM
      Products
  ) AS UniqueProducts
  LEFT JOIN (
    SELECT DISTINCT
      product_id,
      FIRST_VALUE (new_price) OVER (
        PARTITION BY
          product_id
        ORDER BY
          change_date DESC
      ) AS price
    FROM
      Products
    WHERE
      change_date <= '2019-08-16'
  ) AS LastChangedPrice USING (product_id)


# 585. Investments in 2016
# Table: Insurance
# +-------------+-------+
# | Column Name | Type  |
# +-------------+-------+
# | pid         | int   |
# | tiv_2015    | float |
# | tiv_2016    | float |
# | lat         | float |
# | lon         | float |
# +-------------+-------+
# pid is the primary key (column with unique values) for this table.
# Each row of this table contains information about one policy where:
# pid is the policyholder's policy ID.
# tiv_2015 is the total investment value in 2015 and tiv_2016 is the total investment value in 2016.
# lat is the latitude of the policy holder's city. It's guaranteed that lat is not NULL.
# lon is the longitude of the policy holder's city. It's guaranteed that lon is not NULL.
 
# Write a solution to report the sum of all total investment values in 2016 tiv_2016, for all policyholders who:
# have the same tiv_2015 value as one or more other policyholders, and
# are not located in the same city as any other policyholder (i.e., the (lat, lon) attribute pairs must be unique).
# Round tiv_2016 to two decimal places.
# The result format is in the following example.
# Example 1:
# Input: 
# Insurance table:
# +-----+----------+----------+-----+-----+
# | pid | tiv_2015 | tiv_2016 | lat | lon |
# +-----+----------+----------+-----+-----+
# | 1   | 10       | 5        | 10  | 10  |
# | 2   | 20       | 20       | 20  | 20  |
# | 3   | 10       | 30       | 20  | 20  |
# | 4   | 10       | 40       | 40  | 40  |
# +-----+----------+----------+-----+-----+
# Output: 
# +----------+
# | tiv_2016 |
# +----------+
# | 45.00    |
# +----------+
# Explanation: 
# The first record in the table, like the last record, meets both of the two criteria.
# The tiv_2015 value 10 is the same as the third and fourth records, and its location is unique.
# The second record does not meet any of the two criteria. Its tiv_2015 is not like any other policyholders and its location is the same as the third record, which makes the third record fail, too.
# So, the result is the sum of tiv_2016 of the first and last record, which is 45.
SELECT ROUND(SUM(tiv_2016), 2) AS tiv_2016
FROM (
   SELECT *,
       COUNT(*)OVER(PARTITION BY tiv_2015) AS tiv_2015_cnt,
       COUNT(*)OVER(PARTITION BY lat, lon) AS loc_cnt
   FROM Insurance
   )t0
WHERE tiv_2015_cnt > 1
AND loc_cnt = 1


# 1890. The Latest Login in 2020
# Table: Logins
# +----------------+----------+
# | Column Name    | Type     |
# +----------------+----------+
# | user_id        | int      |
# | time_stamp     | datetime |
# +----------------+----------+
# (user_id, time_stamp) is the primary key (combination of columns with unique values) for this table.
# Each row contains information about the login time for the user with ID user_id.
 
# Write a solution to report the latest login for all users in the year 2020. Do not include the users who did not login in 2020.
# Return the result table in any order.
# The result format is in the following example.
# Example 1:
# Input: 
# Logins table:
# +---------+---------------------+
# | user_id | time_stamp          |
# +---------+---------------------+
# | 6       | 2020-06-30 15:06:07 |
# | 6       | 2021-04-21 14:06:06 |
# | 6       | 2019-03-07 00:18:15 |
# | 8       | 2020-02-01 05:10:53 |
# | 8       | 2020-12-30 00:46:50 |
# | 2       | 2020-01-16 02:49:50 |
# | 2       | 2019-08-25 07:59:08 |
# | 14      | 2019-07-14 09:00:00 |
# | 14      | 2021-01-06 11:59:59 |
# +---------+---------------------+
# Output: 
# +---------+---------------------+
# | user_id | last_stamp          |
# +---------+---------------------+
# | 6       | 2020-06-30 15:06:07 |
# | 8       | 2020-12-30 00:46:50 |
# | 2       | 2020-01-16 02:49:50 |
# +---------+---------------------+
# Explanation: 
# User 6 logged into their account 3 times but only once in 2020, so we include this login in the result table.
# User 8 logged into their account 2 times in 2020, once in February and once in December. We include only the latest one (December) in the result table.
# User 2 logged into their account 2 times but only once in 2020, so we include this login in the result table.
# User 14 did not login in 2020, so we do not include them in the result table.
SELECT
    DISTINCT user_id,
    FIRST_VALUE(time_stamp)OVER(PARTITION BY user_id ORDER BY time_stamp DESC) AS last_stamp
FROM
    Logins
WHERE EXTRACT(Year FROM time_stamp) = 2020;