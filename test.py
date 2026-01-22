import base64, requests, sys
img_path = sys.argv[1] if len(sys.argv)>1 else "./scratch/assets/1b36fec6ff336217eddf35c734b629ae0cb30f8cb303b952300f7a61c395bc44/images/1b36fec6ff336217eddf35c734b629ae0cb30f8cb303b952300f7a61c395bc44_p518_imgb64292a08fd52580.png"
b64 = base64.b64encode(open(img_path,'rb').read()).decode()
payload = {"images_b64":[b64]}
r = requests.post("http://211.184.184.238:8008/embed", json=payload, timeout=60)
print("status", r.status_code)
print(r.text[:2000])