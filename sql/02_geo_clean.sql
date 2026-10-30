-- =============================================================================
-- Script: sql/02_geo_clean.sql
-- Database: olist.duckdb
-- Stage: Step 10 - Clean Location Data
-- Purpose: Aggregate raw geolocation coordinates into unique centroids per zip code prefix
-- =============================================================================

-- Ensure table/view alias 'geolocation' is available from raw 'geo'
CREATE OR REPLACE VIEW geolocation AS 
SELECT * FROM geo;

-- -----------------------------------------------------------------------------
-- Step 10 Implementation: Create geo_zip table
-- Reduces 1,000,163 raw geolocation rows to 19,015 unique zip code centroids
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE geo_zip AS
SELECT 
    geolocation_zip_code_prefix AS zip,
    AVG(geolocation_lat) AS lat,
    AVG(geolocation_lng) AS lng
FROM geolocation
GROUP BY 1;

-- -----------------------------------------------------------------------------
-- Verification & Data Hygiene Diagnostics
-- -----------------------------------------------------------------------------

-- 1. Check uniqueness and total record count
SELECT 
    COUNT(*) AS total_rows,
    COUNT(DISTINCT zip) AS unique_zips,
    MIN(lat) AS min_lat,
    MAX(lat) AS max_lat,
    MIN(lng) AS min_lng,
    MAX(lng) AS max_lng
FROM geo_zip;

-- 2. Inspect sample records
SELECT zip, lat, lng 
FROM geo_zip 
ORDER BY zip 
LIMIT 5;

-- 3. Match coverage with customers and sellers
SELECT 
    'Customers Coverage' AS entity,
    COUNT(DISTINCT c.customer_zip_code_prefix) AS entity_unique_zips,
    COUNT(DISTINCT g.zip) AS matched_zips,
    ROUND(COUNT(DISTINCT g.zip) * 100.0 / COUNT(DISTINCT c.customer_zip_code_prefix), 2) AS match_pct
FROM customers c
LEFT JOIN geo_zip g ON c.customer_zip_code_prefix = g.zip
UNION ALL
SELECT 
    'Sellers Coverage',
    COUNT(DISTINCT s.seller_zip_code_prefix),
    COUNT(DISTINCT g.zip),
    ROUND(COUNT(DISTINCT g.zip) * 100.0 / COUNT(DISTINCT s.seller_zip_code_prefix), 2)
FROM sellers s
LEFT JOIN geo_zip g ON s.seller_zip_code_prefix = g.zip;
