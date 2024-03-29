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
SELECT
    DISTINCT user_id,
    FIRST_VALUE(time_stamp)OVER(PARTITION BY user_id ORDER BY time_stamp DESC) AS last_stamp
FROM
    Logins
WHERE EXTRACT(Year FROM time_stamp) = 2020;
