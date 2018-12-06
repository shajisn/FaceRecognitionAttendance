from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, create_engine, ForeignKey
from sqlalchemy.schema import FetchedValue
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

POSTGRES = {
    'user': 'postgres',
    'pw': 'ease@inapp1',
    'db': 'facerecognition',
    'host': 'localhost',
    'port': '5432',
}
dbUrl = 'postgresql://%(user)s:%(pw)s@%(host)s:%(port)s/%(db)s' % POSTGRES

engine = create_engine(dbUrl, convert_unicode=True)
db_session = scoped_session(sessionmaker(autocommit=False,
                                         autoflush=False,
                                         bind=engine))

Base = declarative_base()
Base.query = db_session.query_property()
metadata = Base.metadata


class User(Base):
    __tablename__ = 'users'

    id = Column("user_id", Integer, primary_key=True, server_default=FetchedValue())
    userID = Column("user_name", String(20), nullable=False, unique=True)
    password = Column(String(10))
    email = Column(String(50), unique=True)
    name = Column(String(200))
    registered_on = Column(DateTime)

    def __init__(self, user_id, pswd, email_id, user_name):
        self.userID = user_id
        self.password = pswd
        self.name = user_name
        self.email = email_id
        self.registered_on = datetime.utcnow()

class Attendance(Base):
    __tablename__ = 'attendance'

    id = Column(Integer, primary_key=True, server_default=FetchedValue())
    user = Column(Integer, ForeignKey('users.user_id'))
    logged_on = Column(DateTime)
    logged_out = Column(DateTime)

    def __init__(self, user_id, logged_on):
        self.userID = user_id
        self.logged_on = logged_on

