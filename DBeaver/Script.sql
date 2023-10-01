--Rata-rata umur customer berdasarkan marital status
SELECT 
    CASE 
        WHEN "Marital Status" = '' THEN 'Unknown'
        ELSE "Marital Status"
    END AS marital_status,
    AVG(age) AS average_age
FROM customer
GROUP BY marital_status;

--Rata-rata umur customer berdasarkan gender
SELECT 
	CASE
		WHEN gender = 1 THEN 'male'
		WHEN gender = 0 THEN 'female'
	end AS gender,
	AVG(age) as average_age
FROM customer  
GROUP BY gender; 

--Store dengan total quantity terbanyak
SELECT store.storename, sum(public."Transaction".qty) AS total_qty
FROM store 
JOIN public."Transaction" ON public."Transaction" .storeid = store.storeid
GROUP BY store.storename 
ORDER BY total_qty DESC


--Produk terlaris dengan total amount terbanyak
SELECT product."Product Name" , SUM(public."Transaction".totalamount) AS total_amount
FROM product
JOIN public."Transaction" ON product.productid = public."Transaction" .productid 
GROUP BY product."Product Name" 
ORDER BY total_amount DESC
LIMIT 1;



