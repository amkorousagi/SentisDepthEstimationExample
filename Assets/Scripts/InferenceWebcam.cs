using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using Unity.Sentis;

public class InferenceWebcam : MonoBehaviour
{
    public ModelAsset estimationModel;
    IWorker m_engineEstimation;
    IWorker m_engineEstimationCPU;

    WebCamTexture webcamTexture;
    TensorFloat inputTensor;
    RenderTexture outputTexture;

    public Material material;
    public Texture2D colorMap;

    int modelLayerCount = 0;
    public int framesToExectute = 2;
    private Stopwatch _stopwatch;
    private Stopwatch _stopwatch2;
    
    void Start()
    {
        Application.targetFrameRate = 60;
        var model = ModelLoader.Load(estimationModel);
        var output = model.outputs[0];
        model.layers.Add(new Unity.Sentis.Layers.ReduceMax("max0", new[] { output }, false));
        model.layers.Add(new Unity.Sentis.Layers.ReduceMin("min0", new[] { output }, false));
        model.layers.Add(new Unity.Sentis.Layers.Sub("maxO - minO", "max0", "min0"));
        model.layers.Add(new Unity.Sentis.Layers.Sub("output - min0", output, "min0"));
        model.layers.Add(new Unity.Sentis.Layers.Div("output2", "output - min0", "maxO - minO"));
        modelLayerCount = model.layers.Count;
        model.outputs = new List<string>() { "output2" };
        m_engineEstimation = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
        m_engineEstimationCPU = WorkerFactory.CreateWorker(BackendType.CPU, model);


        WebCamDevice[] devices = WebCamTexture.devices;
        webcamTexture = new WebCamTexture(1920, 1080);
        webcamTexture.deviceName = devices[0].name;
        webcamTexture.Play();

        outputTexture = new RenderTexture(256, 256, 0, RenderTextureFormat.ARGBFloat);
        inputTensor = TensorFloat.Zeros(new TensorShape(1, 3, 256, 256));

        _stopwatch = new Stopwatch();
        _stopwatch2 = new Stopwatch();
        StartCoroutine(CPU());
        StartCoroutine(GPU());

    }

    bool executionStarted = false;
    IEnumerator executionSchedule;

    IEnumerator GPU()
    {
        while (true)
        {
            _stopwatch.Restart();
            TextureConverter.ToTensor(webcamTexture, inputTensor, new TextureTransform());
        
            m_engineEstimation.Execute(inputTensor);
            var output = m_engineEstimation.PeekOutput() as TensorFloat;
            output.MakeReadable();
            output.ToReadOnlyArray();
            _stopwatch.Stop();
            print("infer time gpu : " + _stopwatch.ElapsedMilliseconds+"ms");
            yield return null;
        }
    }
    
    IEnumerator CPU()
    {
        while (true)
        {
            _stopwatch2.Restart();
            TextureConverter.ToTensor(webcamTexture, inputTensor, new TextureTransform());
        
            m_engineEstimationCPU.Execute(inputTensor);
            var output2 = m_engineEstimationCPU.PeekOutput() as TensorFloat;
            output2.MakeReadable();
            output2.ToReadOnlyArray();
            _stopwatch2.Stop();
            print("infer time cpu : " + _stopwatch2.ElapsedMilliseconds+"ms");
            yield return null;

        }
    }

    

    void OnRenderObject()
    {
        material.SetVector("ScreenCamResolution", new Vector4(Screen.height, Screen.width, 0, 0));
        material.SetTexture("WebCamTex", webcamTexture);
        material.SetTexture("DepthTex", outputTexture);
        material.SetTexture("ColorRampTex", colorMap);
        Graphics.Blit(null, material);
    }

    private void OnDestroy()
    {
        m_engineEstimation.Dispose();
        inputTensor.Dispose();
        outputTexture.Release();
    }
}
