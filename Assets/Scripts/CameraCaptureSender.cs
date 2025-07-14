using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using TMPro;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using Unity.Collections;
using static UnityEngine.XR.ARSubsystems.XRCpuImage;

public class ARCamCaptureSender : MonoBehaviour
{
    public string serverUrl = "https://192.168.0.103:8000/explain";

    private ARCameraManager cameraManager;
    private Texture2D texture;
    private RawImage cameraPreview;
    private RectTransform previewBGRT;
    private Button captureButton;
    private TextMeshProUGUI labelText;

    private int rotateMode = 0;
    private bool isFlipped = false;

    void Start()
    {
        cameraManager = FindObjectOfType<ARCameraManager>();
        if (cameraManager == null)
        {
            Debug.LogError("❌ ARCameraManager not found in scene.");
            return;
        }

        CreateUI();
        captureButton.onClick.AddListener(() => StartCoroutine(CaptureAndSendImage()));
    }

    void CreateUI()
    {
        // Canvas
        GameObject canvasGO = new GameObject("UICanvas");
        Canvas canvas = canvasGO.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvasGO.AddComponent<CanvasScaler>().uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        canvasGO.AddComponent<GraphicRaycaster>();

        // Preview Border
        GameObject previewBG = new GameObject("CameraPreviewBG");
        previewBG.transform.SetParent(canvasGO.transform, false);
        Image borderImage = previewBG.AddComponent<Image>();
        borderImage.color = Color.white;

        previewBGRT = previewBG.GetComponent<RectTransform>();
        previewBGRT.sizeDelta = new Vector2(510, 310);
        previewBGRT.anchorMin = previewBGRT.anchorMax = new Vector2(0.5f, 0);
        previewBGRT.anchoredPosition = new Vector2(-100, 160); // Moved slightly left

        // Camera Preview
        GameObject previewGO = new GameObject("CameraPreview");
        previewGO.transform.SetParent(previewBG.transform, false);
        cameraPreview = previewGO.AddComponent<RawImage>();
        cameraPreview.color = Color.white;
        cameraPreview.uvRect = new Rect(0, 0, 1, 1); // Start with no flip

        RectTransform previewRT = cameraPreview.GetComponent<RectTransform>();
        previewRT.sizeDelta = new Vector2(500, 300);
        previewRT.anchorMin = previewRT.anchorMax = new Vector2(0.5f, 0.5f);
        previewRT.anchoredPosition = Vector2.zero;

        // Label Panel
        GameObject panelGO = new GameObject("LabelPanel");
        panelGO.transform.SetParent(canvasGO.transform, false);
        Image panelImage = panelGO.AddComponent<Image>();
        panelImage.color = new Color(1, 1, 1, 0.85f);

        RectTransform panelRT = panelGO.GetComponent<RectTransform>();
        panelRT.anchorMin = new Vector2(0.25f, 0.85f);
        panelRT.anchorMax = new Vector2(0.75f, 0.98f);
        panelRT.offsetMin = panelRT.offsetMax = Vector2.zero;

        GameObject textGO = new GameObject("LabelText");
        textGO.transform.SetParent(panelGO.transform, false);
        labelText = textGO.AddComponent<TextMeshProUGUI>();
        labelText.text = "Waiting for label...";
        labelText.fontSize = 36;
        labelText.color = Color.black;
        labelText.alignment = TextAlignmentOptions.Center;

        RectTransform textRT = labelText.GetComponent<RectTransform>();
        textRT.anchorMin = Vector2.zero;
        textRT.anchorMax = Vector2.one;
        textRT.offsetMin = textRT.offsetMax = Vector2.zero;

        // Capture Button
        GameObject buttonGO = new GameObject("CaptureButton");
        buttonGO.transform.SetParent(canvasGO.transform, false);
        captureButton = buttonGO.AddComponent<Button>();
        Image buttonImage = buttonGO.AddComponent<Image>();
        buttonImage.color = new Color(0.2f, 0.5f, 1f, 1f);

        RectTransform btnRT = buttonGO.GetComponent<RectTransform>();
        btnRT.sizeDelta = new Vector2(250, 80);
        btnRT.anchoredPosition = new Vector2(0, -250);
        btnRT.anchorMin = btnRT.anchorMax = new Vector2(0.5f, 0.5f);

        GameObject btnTextGO = new GameObject("ButtonText");
        btnTextGO.transform.SetParent(buttonGO.transform, false);
        TextMeshProUGUI btnText = btnTextGO.AddComponent<TextMeshProUGUI>();
        btnText.text = "Detect Object";
        btnText.fontSize = 28;
        btnText.color = Color.white;
        btnText.alignment = TextAlignmentOptions.Center;

        RectTransform btnTextRT = btnText.GetComponent<RectTransform>();
        btnTextRT.anchorMin = Vector2.zero;
        btnTextRT.anchorMax = Vector2.one;
        btnTextRT.offsetMin = btnTextRT.offsetMax = Vector2.zero;

        // Rotate Button
        CreateSimpleButton(canvasGO.transform, "Rotate", new Vector2(0.85f, 0.15f), () =>
        {
            rotateMode = (rotateMode + 1) % 4;
            float angle = rotateMode * 90f;
            previewBGRT.localEulerAngles = new Vector3(0, 0, angle);
        });

        // Flip Button
        CreateSimpleButton(canvasGO.transform, "Flip", new Vector2(0.85f, 0.05f), () =>
        {
            isFlipped = !isFlipped;
            cameraPreview.uvRect = isFlipped
                ? new Rect(0, 1, 1, -1)
                : new Rect(0, 0, 1, 1);
        });
    }

    void CreateSimpleButton(Transform parent, string label, Vector2 anchor, UnityEngine.Events.UnityAction action)
    {
        GameObject btnGO = new GameObject(label + "Button");
        btnGO.transform.SetParent(parent, false);
        Button button = btnGO.AddComponent<Button>();
        Image image = btnGO.AddComponent<Image>();
        image.color = new Color(0.3f, 0.3f, 0.3f, 1f);

        RectTransform rt = btnGO.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(140, 50);
        rt.anchorMin = rt.anchorMax = anchor;
        rt.anchoredPosition = Vector2.zero;

        GameObject txtGO = new GameObject(label + "Text");
        txtGO.transform.SetParent(btnGO.transform, false);
        TextMeshProUGUI txt = txtGO.AddComponent<TextMeshProUGUI>();
        txt.text = label;
        txt.fontSize = 20;
        txt.color = Color.white;
        txt.alignment = TextAlignmentOptions.Center;

        RectTransform txtRT = txt.GetComponent<RectTransform>();
        txtRT.anchorMin = Vector2.zero;
        txtRT.anchorMax = Vector2.one;
        txtRT.offsetMin = txtRT.offsetMax = Vector2.zero;

        button.onClick.AddListener(action);
    }

    IEnumerator CaptureAndSendImage()
    {
        if (cameraManager == null || !cameraManager.TryAcquireLatestCpuImage(out XRCpuImage cpuImage))
        {
            Debug.LogWarning("⚠️ Could not acquire camera image.");
            labelText.text = "Camera not ready";
            yield break;
        }

        using (cpuImage)
        {
            var conversionParams = new XRCpuImage.ConversionParams
            {
                inputRect = new RectInt(0, 0, cpuImage.width, cpuImage.height),
                outputDimensions = new Vector2Int(cpuImage.width, cpuImage.height),
                outputFormat = TextureFormat.RGB24,
                transformation = Transformation.None
            };

            var rawTextureData = new NativeArray<byte>(cpuImage.GetConvertedDataSize(conversionParams), Allocator.Temp);
            cpuImage.Convert(conversionParams, rawTextureData);

            if (texture == null || texture.width != conversionParams.outputDimensions.x || texture.height != conversionParams.outputDimensions.y)
            {
                texture = new Texture2D(conversionParams.outputDimensions.x, conversionParams.outputDimensions.y, TextureFormat.RGB24, false);
            }

            texture.LoadRawTextureData(rawTextureData);
            texture.Apply();
            rawTextureData.Dispose();

            cameraPreview.texture = texture;

            byte[] imageBytes = texture.EncodeToJPG();
            string base64Image = System.Convert.ToBase64String(imageBytes);
            string json = JsonUtility.ToJson(new ImagePayload { image = base64Image });

            UnityWebRequest request = new UnityWebRequest(serverUrl, "POST");
            request.uploadHandler = new UploadHandlerRaw(System.Text.Encoding.UTF8.GetBytes(json));
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.certificateHandler = new BypassCertificate();

            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("❌ Server error: " + request.error);
                labelText.text = "Detection failed";
            }
            else
            {
                HandleResponse(request.downloadHandler.text);
            }
        }
    }

    void HandleResponse(string responseText)
    {
        try
        {
            ExplainResponse response = JsonUtility.FromJson<ExplainResponse>(responseText);
            Debug.Log($"✅ Prediction: {response.label} — {response.explanation}");
            labelText.text = $"Detected: {response.label}";
        }
        catch
        {
            Debug.LogError("❌ Failed to parse server response");
            labelText.text = "Invalid server response";
        }
    }

    [System.Serializable]
    public class ImagePayload
    {
        public string image;
    }

    [System.Serializable]
    public class ExplainResponse
    {
        public string label;
        public string explanation;
    }

    private class BypassCertificate : CertificateHandler
    {
        protected override bool ValidateCertificate(byte[] certificateData) => true;
    }
}
