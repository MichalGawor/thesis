import os
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart


username = ""
password = ""
_from = ""
_to = ""

def send_mail(image_path):
    image = open(image_path, 'rb').read()

    message = MIMEMultipart()
    message['Subject'] = 'subject'
    message['From'] = 'e@mail.cc'
    message['To'] = 'e@mail.cc'

    text = MIMEText("test")
    message.attach(text)
    image = MIMEImage(image, name=os.path.basename(image_path))
    message.attach(image)

    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(username, password)
    server.sendmail(_from, _to, message.as_string())
    server.quit()

