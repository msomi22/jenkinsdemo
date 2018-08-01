



-- Schema Name: quickquerydb
-- Username: quickquery
-- Password: qq@future 


\c postgres

-- Then execute the following:
DROP DATABASE IF EXISTS quickquerydb; 
CREATE DATABASE quickquerydb;

GRANT ALL PRIVILEGES ON DATABASE  quickquerydb to  quickquery;
ALTER DATABASE quickquerydb OWNER to quickquery;

-- Connect with the database on the username
\c quickquerydb quickquery



-- =========================
-- 1. Account Management
-- =========================


--------------------
-- Table Account
-- -------------------
CREATE TABLE  Account (
    id SERIAL PRIMARY KEY,
    uuid VARCHAR(255) UNIQUE NOT NULL,
    is_active BOOLEAN NOT NULL, 
    business_name VARCHAR(255) NOT NULL, 
    business_website VARCHAR(255) NOT NULL,
    business_mobile VARCHAR(15) NOT NULL, 
    business_email VARCHAR(100) NOT NULL,
    business_address VARCHAR(100) NOT NULL,
    business_town VARCHAR(100) NOT NULL,
    last_updated VARCHAR(255) NOT NULL,
    creation_date timestamp with time zone DEFAULT now()

);
\COPY Account(uuid,is_active,business_name,business_website,business_mobile,business_email,business_address,business_town,last_updated) FROM '/tmp/Account.csv' WITH DELIMITER AS '|' CSV HEADER
ALTER TABLE Account OWNER TO quickquery;




-- -------------------
-- Table Product
-- -------------------

CREATE TABLE  Product (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR(255) REFERENCES Account(uuid),
    category VARCHAR(255) NOT NULL,
    sub_category VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    model VARCHAR(100) NOT NULL, 
    unit INTEGER NOT NULL,
    price FLOAT(53),   
    stock_count INTEGER NOT NULL,
    last_updated VARCHAR(255) NOT NULL,
    add_date timestamp with time zone DEFAULT now()
);
\COPY Product(account_id,category,sub_category,name,model,unit,price,stock_count,last_updated) FROM '/tmp/Product.csv' WITH DELIMITER AS '|' CSV HEADER
ALTER TABLE Product OWNER TO quickquery;


