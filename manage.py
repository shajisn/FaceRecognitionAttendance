from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand
from app import app
from flask_sqlalchemy  import SQLAlchemy
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, ForeignKey


db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = 'users'

    id = Column("user_id", Integer, primary_key=True)
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

class Attendance(db.Model):
    __tablename__ = 'attendance'

    id = Column(Integer, primary_key=True)
    user = Column(Integer, ForeignKey('users.user_id'))
    logged_on = Column(DateTime)
    logged_out = Column(DateTime)


migrate = Migrate(app, db)
manager = Manager(app)

manager.add_command('db', MigrateCommand)


if __name__ == '__main__':
    manager.run()