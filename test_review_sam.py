"""SAM API contracts without GPU or writes to real datasets."""
import base64
import json
from http.server import ThreadingHTTPServer
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from review_gui import make_handler
from review_sam import SamError, SamService, validate_points
import test_review_gui


class PointTests(unittest.TestCase):
    def test_points_validate_pixel_bounds_labels_and_duplicates(self):
        p={'x':10,'y':20,'label':1}
        self.assertEqual(len(validate_points([p,p],48,32)),1)
        for points in ([],[{'x':1,'y':1,'label':0}],[dict(p,x=float('nan'))],[dict(p,x=48)],
                       [dict(p,y=-1)],[dict(p,label=True)],[dict(p,label=2)], [p,dict(p,label=0)], [p]*65):
            with self.assertRaises(SamError):validate_points(points,48,32)

    def test_unavailable_and_busy_are_explicit(self):
        service=SamService(python='/missing',checkpoint='/missing')
        self.assertFalse(service.info()['available'])
        with self.assertRaises(SamError):service.predict('/unused',48,32,[{'x':10,'y':10,'label':1}])
        with service.lock:
            with self.assertRaises(SamError) as error:service.predict('/unused',48,32,[{'x':10,'y':10,'label':1}])
            self.assertEqual(error.exception.status,409)

    def test_api_auth_validation_and_read_only_inference(self):
        fixture=test_review_gui.ReviewTests();fixture.setUp()
        class FakeSam:
            calls=[]
            def info(self):return {'available':True}
            def predict(self,path,w,h,points):
                self.calls.append((path,w,h,points))
                return {'width':w,'height':h,'mask':base64.b64encode(bytes([1])*w*h).decode(),'prompt_violations':0}
        fake=FakeSam();server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(fixture.store,fake))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url=f'http://127.0.0.1:{server.server_port}/api/frame/{fixture.frame_id}/sam'
        before=fixture.store.current(fixture.store.frame(fixture.frame_id))
        try:
            data=json.dumps({'points':[{'x':10,'y':10,'label':1}],'image_path':'/not/allowed'}).encode()
            request=Request(url,data=data,headers={'Content-Type':'application/json'})
            with self.assertRaises(HTTPError) as error:urlopen(request)
            self.assertEqual(error.exception.code,403)
            request.add_header('X-Review-Token',fixture.store.token)
            with urlopen(request) as r:self.assertEqual(json.load(r)['width'],48)
            self.assertEqual(fake.calls[0][0],fixture.root/'image.png')
            self.assertEqual(len(fake.calls),1)
            bad=Request(url,data=b'{"points":[]}',headers={'Content-Type':'application/json','X-Review-Token':fixture.store.token})
            with self.assertRaises(HTTPError) as error:urlopen(bad)
            self.assertEqual(error.exception.code,400)
            self.assertEqual(len(fake.calls),1)
            self.assertEqual(before,fixture.store.current(fixture.store.frame(fixture.frame_id)))
            self.assertFalse((fixture.pseudo/'_review_history').exists())
        finally:
            server.shutdown();server.server_close();thread.join();fixture.tearDown()


if __name__=='__main__':unittest.main()
