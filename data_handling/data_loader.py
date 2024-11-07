from pymongo import MongoClient
import pandas as pd
import os
from dotenv import load_dotenv
import csv
from datetime import datetime
import certifi
import logging
import ssl
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("pymongo")

load_dotenv()

MONGODB_URL = os.getenv('MONGODB_URL')
DB_NAME = os.getenv('DB_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

def load_data_from_mongodb(limit=None, batch_size=1500):
    client = None
    try:
        # Create TLS/SSL context
        tls_context = ssl.create_default_context(cafile=certifi.where())
        tls_context.minimum_version = ssl.TLSVersion.TLSv1_2
        
        # Updated MongoDB client configuration with current parameters
        client = MongoClient(
            MONGODB_URL,
            tls=True,
            tlsCAFile=certifi.where(),
            tlsAllowInvalidCertificates=False,  # Enforce certificate validation
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=20000,
            socketTimeoutMS=20000,
            maxPoolSize=1,
            retryWrites=True,
            w='majority'
        )

        # Test connection with timeout
        try:
            client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
        except ConnectionFailure as e:
            logger.error(f"Server not available: {str(e)}")
            raise
        except ServerSelectionTimeoutError as e:
            logger.error(f"Server selection timeout: {str(e)}")
            raise

        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        try:
            # Use find with batch processing
            cursor = collection.find(
                {},  # query
                batch_size=batch_size
            ).batch_size(batch_size)

            if limit:
                cursor = cursor.limit(limit)

            # Process documents in batches
            batch = []
            for document in cursor:
                batch.append(document)
                if len(batch) >= batch_size:
                    for doc in batch:
                        yield doc
                    batch = []
                    logger.debug(f"Processed batch of {batch_size} documents")

            # Yield remaining documents
            for doc in batch:
                yield doc

            logger.info("Successfully completed data retrieval")

        except Exception as e:
            logger.error(f"Error while fetching data: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"MongoDB connection error: {str(e)}")
        raise

    finally:
        if client:
            try:
                client.close()
                logger.info("MongoDB connection closed")
            except Exception as e:
                logger.error(f"Error while closing connection: {str(e)}")