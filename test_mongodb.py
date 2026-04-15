from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi

uri = "mongodb://jahanvigarg238_db_user:N0bHKRnWLUPY3wob@ac-gsttplp-shard-00-00.wn0cz4w.mongodb.net:27017,ac-gsttplp-shard-00-01.wn0cz4w.mongodb.net:27017,ac-gsttplp-shard-00-02.wn0cz4w.mongodb.net:27017/?replicaSet=atlas-9kvd83-shard-0&authSource=admin&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(
    uri,
    tls=True,                        # use tls= instead of ssl= in URI
    tlsCAFile=certifi.where(),       # provides correct CA certs on Windows
    server_api=ServerApi('1')        # keeps Atlas API version stable
)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)