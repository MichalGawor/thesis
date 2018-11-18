import os
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart


__model_mail__ = ""
__password__ = ""
_from = ""
_to = ""

def send_mail(image_path):
    image = open(image_path, 'rb').read()

    message = MIMEMultipart()
    message['Subject'] = 'Result of your artifical painting'
    message['From'] = _from
    message['To'] = _to

    text = MIMEText("Hello!\n\n You can find the result of your artifical painting order attached to this email\n\n Best regards,\nArtificalArt")
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

