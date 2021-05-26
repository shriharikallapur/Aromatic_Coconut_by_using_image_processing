const video = document.getElementById('video');
const canvas = document.getElementById("canvas");
const snap = document.getElementById("snap");
const errorMsgElement = document.getElementById("spanErrorMsg"); 

const constrains = {
    video: {
      width:600, height:400
    },
    audio: false,
};

async function init(){
    try{
        const stream =await navigator.mediaDevices.getUserMedia(constrains);
        handleSuccess(stream);
    }
    catch(e){
      errorMsgElement.innerHTML = 'navigator.getUserMedia.error:$error:${e.toString()}}'
    }
}

function handleSuccess(stream){
  window.stream = stream;
  video.srcObject = stream;
}
init();
var context = canvas.getContext("2d");
snap.addEventListener("click", function(){
    context.drawImage(video, 0, 0, 640, 480);
});

