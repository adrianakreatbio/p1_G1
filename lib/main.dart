import 'dart:io';
import 'dart:typed_data';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:camera/camera.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:open_filex/open_filex.dart';



//--- 1 - INITIAL APP SETUP
// 1 - 1 - SET MAIN SYSTEM
Future<void> main() async {                       // runs the main fxn asynchronously, so it can await for setup steps like flutter binding & finding cameras
  WidgetsFlutterBinding.ensureInitialized();      // initialize flutter - connect flutter framework to feature app         
  final cameras = await availableCameras();       // final means can only use this variable camera once - find available cameras
  runApp(GelApp(cameras: cameras));               // run the app: calling GelApp to build the software  
}

// 1 - 2 - CREATE STATELESS WIDGET AS BASE - GelApp (stateless widget = constant app, no changes throughout)
class GelApp extends StatelessWidget {
  final List<CameraDescription> cameras;              // cameras - stores list of cameras
  const GelApp({super.key, required this.cameras});   // pass camera lists when running GelApp

  @override
  Widget build(BuildContext context) {
    return MaterialApp(title: 'Measure Gel DNA', theme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.blue,), home: GelHomePage(cameras: cameras),);
  } // System-level app name if the app does not have a title (set by AppBar(title:...))
}

// 1 - 3 - CREATE STATEFUL WIDGET - GelHomePage
class GelHomePage extends StatefulWidget {
  final List<CameraDescription> cameras;
  const GelHomePage({super.key, required this.cameras});

  @override
  State<GelHomePage> createState() => _GelHomePageState();  // Class _GelHomePageState will control this stateful widget GelHomePage
}

// 1 - 4 - CREATE STATE THAT CONTROLS THE 3 TABS
class _GelHomePageState extends State<GelHomePage> {
  int _selectedIndex = 0;
  final GlobalKey<_MainPageState> _mainPageKey = GlobalKey<_MainPageState>();               // _GelHomePageState control parent tab view: Access to mainPage & AnalysisPage so GelHomePage can call methods inside the 2 pages
  final GlobalKey<_AnalysisPageState> _analysisPageKey = GlobalKey<_AnalysisPageState>();

// 1 - 4 - 1 ACTION - Warning before swapping tabs (Discard or Save the image)
  Future<void> _onTabChange(int index) async {
    if (_selectedIndex == 0 && index != 0) {
      final mainPageState = _mainPageKey.currentState;
      if (mainPageState?.capturedImage != null && mainPageState?.tempImageFile != null) {
        if (!mounted) return;
        final result = await showDialog<bool>(
          context: context,
          builder: (context) => AlertDialog(
            title: const Text("Unsaved Image"),
            content: const Text("You have an unsaved captured image. Do you want to save it before leaving?"),
            actions: [
              TextButton(onPressed: () => Navigator.pop(context, false), child: const Text("Discard"),),
              TextButton(onPressed: () => Navigator.pop(context, true), child: const Text("Save"),),
            ],
          ),
        );

        if (result == true) {
          await mainPageState?._saveImageToDevice();
        } else if (result == false) {
          if (mainPageState?.tempImageFile != null && mainPageState!.tempImageFile!.existsSync()) {await mainPageState.tempImageFile!.delete();}
          mainPageState?.clearCapturedImage();
        } else {
          return;
        }
      }
    }
    setState(() => _selectedIndex = index);
  }

// 1 - 4 - 2 BUILD THE 3 TABS
  @override
  Widget build(BuildContext context) {
    final screens = [MainPage(cameras: widget.cameras, key: _mainPageKey, analysisPageKey: _analysisPageKey, onGoToResult: () {setState(() {_selectedIndex = 1;});}), AnalysisPage(mainPageKey: _mainPageKey, key: _analysisPageKey,), HistoryPage(key: GlobalKey(), analysisPageKey: _analysisPageKey, onGoToResult: () {setState(() => _selectedIndex = 1);},)];
    return Scaffold(
      appBar: AppBar(title: const Text("Gel DNA Quantification"), elevation: 2,),
      body: Column(
        children: [
          Container(
            color: Colors.grey[100],
            child: Row(
              children: [

                Expanded(
                  child: TextButton(
                    onPressed: () => _onTabChange(0),
                    style: TextButton.styleFrom(backgroundColor: _selectedIndex == 0 ? Colors.blue : Colors.transparent, shape: const RoundedRectangleBorder(borderRadius: BorderRadius.zero),),
                    child: Text("Measure", style: TextStyle(color: _selectedIndex == 0 ? Colors.white : Colors.black, fontWeight: FontWeight.w600,),),
                  ),
                ),

                Expanded(
                  child: TextButton(
                    onPressed: () => _onTabChange(1),
                    style: TextButton.styleFrom(backgroundColor: _selectedIndex == 1 ? Colors.blue : Colors.transparent, shape: const RoundedRectangleBorder(borderRadius: BorderRadius.zero),),
                    child: Text("Result", style: TextStyle(color: _selectedIndex == 1 ? Colors.white : Colors.black, fontWeight: FontWeight.w600,),),
                  ),
                ),

                Expanded(
                  child: TextButton(
                    onPressed: () => _onTabChange(2),
                    style: TextButton.styleFrom(backgroundColor: _selectedIndex == 2 ? Colors.blue : Colors.transparent, shape: const RoundedRectangleBorder(borderRadius: BorderRadius.zero),),
                    child: Text("Saved", style: TextStyle(color: _selectedIndex == 2 ? Colors.white : Colors.black, fontWeight: FontWeight.w600,),),
                  ),
                ),

              ],
            ),
          ),
          Expanded(child: IndexedStack(index: _selectedIndex, children: screens,)),
        ],
      ),
    );
  }
}


//--- 2 - SETUP MAINPAGE: MEASURE (STATEFUL WIDGET, STATE)
class MainPage extends StatefulWidget {
  final List<CameraDescription> cameras;
  final GlobalKey<_AnalysisPageState> analysisPageKey;
  final VoidCallback onGoToResult;                        // this ongotoresult is connected to analysispage
  const MainPage({super.key, required this.cameras, required this.analysisPageKey, required this.onGoToResult,});

  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
  CameraController? _cameraController;
  Future<void>? _initializeControllerFuture;
  late XFile? _capturedImage;
  late File? _tempImageFile;
  final TextEditingController _standardController = TextEditingController();
  bool _isLoading = false;
  final String _analyzeUrl = "http://127.0.0.1:8000/analyze";   // call app - analyze fxn
  final String _analyzeJSONUrl = "http://127.0.0.1:8000/analyze_json";

// 2 - 1 - CREATE GLOBAL ACCESS GETTERS - CAPTURED IMAGE AND TEMPFILE
  XFile? get capturedImage => _capturedImage;
  File? get tempImageFile => _tempImageFile;

  // var: once mainpage is created: empty capturedimage, empty tempimage vars and initiate camera
  @override
  void initState() {
    super.initState();
    _capturedImage = null;
    _tempImageFile = null;
    _initCamera();
  }

  // var: once tab is switched/discarded: clear capturedimage and tempimagefile
  void clearCapturedImage() {
  setState(() {
    _capturedImage = null;
    _tempImageFile = null;
  });
  }

// 2 - 2 - CONNECT CAMERA DEVICE
  Future<void> _initCamera() async {
    if (widget.cameras.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("No camera found")),);
      return;
    }

    final status = await Permission.camera.request();
    if (!status.isGranted) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Camera permission denied")),);
      return;
    }

    // choose camera
    final CameraDescription selected = widget.cameras.firstWhere(
      (cam) {
        final name = cam.name.toLowerCase();
        return name.contains('4k') || name.contains('imx') || name.contains('usb20');
      },
      orElse: () => widget.cameras.first,
    );

    final controller = CameraController(selected, ResolutionPreset.high, enableAudio: false,);

    setState(() {
      _cameraController = controller;
      _initializeControllerFuture = controller.initialize();
    });
  }

// 2 - 3 - ACTION - CAPTURE IMAGE
  Future<void> _captureImage() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Camera not ready")),);
      return;
    }

    // If there's an existing unsaved image, prompt to save/delete first
    if (_capturedImage != null && _tempImageFile != null) {
      if (!mounted) return;
      final result = await showDialog<bool>(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text("Unsaved Image"),
          content: const Text("You have an unsaved captured image. Do you want to save it before capturing a new one?"),
          actions: [TextButton(onPressed: () => Navigator.pop(context, false), child: const Text("Discard"),), TextButton(onPressed: () => Navigator.pop(context, true), child: const Text("Save"),),],
        ),
      );

      if (result == true) {
        await _saveImageToDevice();
      } else if (result == false) {
        if (_tempImageFile != null && _tempImageFile!.existsSync()) {await _tempImageFile!.delete();}
        setState(() {
          _capturedImage = null;
          _tempImageFile = null;
        });
      } else {
        return; // User cancelled
      }
    }

    // take picture
    try {
      await _initializeControllerFuture!;
      final XFile photo = await _cameraController!.takePicture();
      setState(() {_capturedImage = photo;});
      _tempImageFile = File(photo.path);
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Image captured")),);
    } catch (e) {ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Capture error: $e")),);}
  }

// 2 - 4 - ACTION - SAVE IMAGE TO DEVICE
  Future<void> _saveImageToDevice() async {
    if (_capturedImage == null) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("No image to save")),);
      return;
    }

    final status = await Permission.storage.request();
    if (!status.isGranted) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Storage permission denied")),);
      return;
    }

    try {
      final appDir = await getApplicationDocumentsDirectory();
      final imageDir = Directory('${appDir.path}/Saved_images');
      if (!imageDir.existsSync()){imageDir.createSync(recursive: true);}
      final imgName = _capturedImage!.path.split(Platform.pathSeparator).last;
      final nameParts = imgName.split('.');
      final base = nameParts.first;          // "PhotoCapture_2025_1111_145103_492"

      String timePart;
      if (base.startsWith('PhotoCapture_')) {timePart = base.substring('PhotoCapture_'.length);} else {timePart = base;}

      final gelBase = 'Gel_$timePart';
      final gelPath = "${imageDir.path}/$gelBase.png";
      final file = File(gelPath);
      final bytes = await _capturedImage!.readAsBytes();
      await file.writeAsBytes(bytes);
      _tempImageFile = null;  // Clear temporary file reference since it's now saved

      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Image saved: $gelPath")),);} catch (e) {ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Error saving image: $e")),);}
  }

  // 2 - 5 - ACTION - MEASURE DNA CONC
  Future<void> _measureDNA() async {
    // QC
    if (_capturedImage == null) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Capture a gel image first")),);
      return;
    }
    final standardText = _standardController.text.trim();
    if (standardText.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Enter standard concentration for lane 1")),);
      return;
    }
    final standardValue = double.tryParse(standardText);
    if (standardValue == null) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Standard concentration must be numeric")),);
      return;
    }
    final status = await Permission.storage.request();
    if (!status.isGranted) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Storage permission denied")),);
      return;
    }

    setState(() { _isLoading = true; });

    try {
      final imgName = _capturedImage!.path.split(Platform.pathSeparator).last;
      final base = imgName.split('.').first;
      final ext = imgName.split('.').last;
      final timePart = base.startsWith('PhotoCapture_') ? base.substring('PhotoCapture_'.length) : base;

      // ========== STEP 1: CREATE REPORT DIRECTORY ==========
      final dir = await getApplicationDocumentsDirectory();
      final reportRoot = Directory('${dir.path}/Reports');
      if (!reportRoot.existsSync()) reportRoot.createSync(recursive: true);
      final reportDir = Directory('${reportRoot.path}/Data_$timePart');
      if (!reportDir.existsSync()) reportDir.createSync(recursive: true);

      final gelPath = '${reportDir.path}/Gel_$timePart.$ext';
      await File(_capturedImage!.path).copy(gelPath);
      print("DEBUG: Gel image copied to $gelPath");

      // ========== STEP 2: CALL /analyze ENDPOINT ==========
      print("DEBUG: Calling /analyze endpoint...");
      final uri = Uri.parse(_analyzeUrl);
      final request = http.MultipartRequest("POST", uri)
        ..fields["standard_conc"] = standardValue.toString()
        ..files.add(await http.MultipartFile.fromPath("file", gelPath));
      final streamedResponse = await request.send();

      if (streamedResponse.statusCode != 200) {
        setState(() { _isLoading = false; });
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Server error: ${streamedResponse.statusCode}")),);
        return;
      }

      final Uint8List pdfBytes = await streamedResponse.stream.toBytes();
      print("DEBUG: PDF received (${pdfBytes.length} bytes)");

      // ========== STEP 3: SAVE PDF ==========
      final pdfPath = '${reportDir.path}/Report_$timePart.pdf';
      await File(pdfPath).writeAsBytes(pdfBytes);
      print("DEBUG: PDF saved to $pdfPath");

      // ========== STEP 4: DOWNLOAD ANNOTATED IMAGE FROM BACKEND ==========
      print("DEBUG: Downloading annotated image...");
      Uint8List? annotatedBytes;
      try {
        final annotateUri = Uri.parse("http://127.0.0.1:8000/get_annotated/$timePart");
        final annotateResponse = await http.get(annotateUri);
        if (annotateResponse.statusCode == 200) {
          annotatedBytes = annotateResponse.bodyBytes;
          final annotatedPath = '${reportDir.path}/Annotated_$timePart.png';
          await File(annotatedPath).writeAsBytes(annotatedBytes);
          print("DEBUG: ✓ Annotated image saved (${annotatedBytes.length} bytes)");
        } else {
          print("DEBUG: ✗ Could not download annotated image: ${annotateResponse.statusCode}");
        }
      } catch (e) {
        print("DEBUG: ✗ Error downloading annotated: $e");
      }

      // ========== STEP 5: DOWNLOAD MASK IMAGE FROM BACKEND ==========
      print("DEBUG: Downloading mask image...");
      Uint8List? maskBytes;
      try {
        final maskUri = Uri.parse("http://127.0.0.1:8000/get_mask/$timePart");
        final maskResponse = await http.get(maskUri);
        if (maskResponse.statusCode == 200) {
          maskBytes = maskResponse.bodyBytes;
          final maskPath = '${reportDir.path}/Mask_$timePart.png';
          await File(maskPath).writeAsBytes(maskBytes);
          print("DEBUG: ✓ Mask image saved (${maskBytes.length} bytes)");
        } else {
          print("DEBUG: ✗ Could not download mask image: ${maskResponse.statusCode}");
        }
      } catch (e) {
        print("DEBUG: ✗ Error downloading mask: $e");
      }

      // ========== STEP 6: GET LANE CONCENTRATIONS ==========
      print("DEBUG: Getting lane concentrations...");
      List<List<double>>? laneConcs;
      try {
        final jsonUri = Uri.parse(_analyzeJSONUrl);
        final jsonRequest = http.MultipartRequest("POST", jsonUri)
          ..fields["standard_conc"] = standardValue.toString()
          ..files.add(await http.MultipartFile.fromPath("file", gelPath));
        final jsonResponse = await http.Response.fromStream(await jsonRequest.send());
        
        if (jsonResponse.statusCode == 200) {
          final data = jsonDecode(jsonResponse.body) as Map<String, dynamic>;
          final raw = data["lane_concs"] as List;
          laneConcs = raw.map<List<double>>((lane) {
            if (lane is List) {
              return lane.map<double>((e) => (e as num).toDouble()).toList();
            } else if (lane is num) {
              return [lane.toDouble()];
            } else {
              return <double>[];
            }
          }).toList();
          print("DEBUG: ✓ Lane concentrations: $laneConcs");
        }
      } catch (e) {
        print("DEBUG: ✗ Error getting concentrations: $e");
      }

      // ========== STEP 7: UPDATE ANALYSIS PAGE ==========
      final analysisState = widget.analysisPageKey.currentState;
      if (analysisState != null) {
        analysisState.setState(() {
          analysisState._laneConcs = laneConcs;
          analysisState._annotatedBytes = annotatedBytes;
          analysisState._maskBytes = maskBytes;
        });
        print("DEBUG: ✓ Analysis page updated");
      }

      setState(() { _isLoading = false; });
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Analysis complete!")));
      await OpenFilex.open(pdfPath);
      widget.onGoToResult();
      
    } catch (e) {
      print("DEBUG: ✗✗✗ Error: $e");
      setState(() { _isLoading = false; });
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Error: $e")));
    }
  }

  Future<bool> _onWillPop() async {     // If user press back or recapture image, warning will pop out
    if (_capturedImage != null && _tempImageFile != null) {
      final result = await showDialog<bool>(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text("Unsaved Image"),
          content: const Text("You have an unsaved captured image. Do you want to save it before closing?"),
          actions: [TextButton(onPressed: () => Navigator.pop(context, false), child: const Text("Discard"),), TextButton(onPressed: () => Navigator.pop(context, true), child: const Text("Save"),),],
        ),
      );

      if (result == true) {
        await _saveImageToDevice();
        return true;
      } else if (result == false) {
        if (_tempImageFile != null && _tempImageFile!.existsSync()) {await _tempImageFile!.delete();}
        return true;
      }
      return false;
    }
    return true;
  }

  @override
  void dispose() {
    _standardController.dispose();
    _cameraController?.dispose();
    if (_tempImageFile != null && _tempImageFile!.existsSync()) {_tempImageFile!.delete();}   // delete if it is unsaved
    super.dispose();
  }

// 2 - 5 - 6 - BUILD LAYOUT: LEFT (VIDEO), RIGHT (IMAGE)
  @override
  Widget build(BuildContext context) {
    // IMAGE
    final capturedPreview = _capturedImage == null ? const Center(child: Text("No image captured")) : Image.file(File(_capturedImage!.path), fit: BoxFit.cover,);
    // VIDEO
    Widget previewArea;
    if (_cameraController == null || _initializeControllerFuture == null) {
      previewArea = const Center(child: Text("Camera not initialized"));
    } else {
      previewArea = FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {if (snapshot.connectionState == ConnectionState.done) {return CameraPreview(_cameraController!);} else {return const Center(child: CircularProgressIndicator(strokeWidth: 2),);}},
      );
    }
    final lastImageInfo = _capturedImage == null ? const Text("No image captured yet") : Text("Captured image path: ${_capturedImage!.path}", style: const TextStyle(fontSize: 12));

    return WillPopScope(
      onWillPop: _onWillPop,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Expanded(
            child: Row(
              children: [
                Expanded(child: Container(color: Colors.black, child: previewArea,),),
                const SizedBox(width: 12),
                Expanded(child: Container(color: Colors.grey[200], child: capturedPreview,),),
              ],
            ),
          ),
          const SizedBox(height: 8),
          lastImageInfo,
          const SizedBox(height: 16),
          TextField(controller: _standardController, keyboardType: const TextInputType.numberWithOptions(decimal: true), decoration: const InputDecoration(border: OutlineInputBorder(), labelText: "Lane 1 standard DNA concentration (ng/µL)",),),
          const SizedBox(height: 16),
          Wrap(spacing: 8, runSpacing: 8,
            children: [
              ElevatedButton(onPressed: _captureImage, child: const Text("Capture image"),),
              ElevatedButton(onPressed: _saveImageToDevice, child: const Text("Save image"),),
              ElevatedButton(onPressed: _isLoading ? null : _measureDNA,
                child: _isLoading
                    ? const SizedBox(height: 18, width: 18, child: CircularProgressIndicator(strokeWidth: 2),)
                    : const Text("Measure Concentration"),
              ),
            ],
          ),
          const SizedBox(height: 50),
        ],
      ),
    ),);
  }
} // MainPage ends


//--- 3 - SETUP ANALYSIS PAGE: RESULT (STATEFUL WIDGET, STATE)
class AnalysisPage extends StatefulWidget {
  final GlobalKey<_MainPageState> mainPageKey;                
  const AnalysisPage({super.key, required this.mainPageKey});   // call mainpage state

  @override
  State<AnalysisPage> createState() => _AnalysisPageState();    // Create analysis page state
}

class _AnalysisPageState extends State<AnalysisPage> {
  Uint8List? _annotatedBytes;
  Uint8List? _maskBytes;
  bool _isLoading = false;
  String? _error;
  final String _annotateUrl = "http://127.0.0.1:8000/annotate";
  final String _maskUrl = "http://127.0.0.1:8000/mask";


// 3 - 2 - BUILD LAYOUT: LEFT (CONCENTRATION), RIGHT (PNG LABEL, PNG MASK)
  List<List<double>>? _laneConcs;   // DNA Concentration: [[200, 140], [130], ...]
  @override
  Widget build(BuildContext context) {
    final mainPageState = widget.mainPageKey.currentState;
    //final captured = mainPageState?.capturedImage;
    final hasResults = _laneConcs != null || _annotatedBytes != null || _maskBytes != null;

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          const SizedBox(height: 16),
          // MAIN AREA
          if (!hasResults)
            const Expanded(child: Center(child: Text("No analysis done."),),)
          else
            Expanded(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Expanded(   // LEFT: concentration text
                    flex: 1,
                    child: Container(
                      padding: const EdgeInsets.all(8),
                      color: Colors.grey[100],
                      child: _laneConcs == null
                          ? const Center(child: Text("NA", textAlign: TextAlign.center,),)
                          : ListView.builder(
                              itemCount: _laneConcs!.length,
                              itemBuilder: (context, index) {
                                final laneIndex = index + 1;
                                final bands = _laneConcs![index];
                                final bandsText = bands.isEmpty ? "-" : List.generate(bands.length, (i) => "${bands[i].toStringAsFixed(1)} (band ${i + 1})",).join(", ");
                                return Padding(padding: const EdgeInsets.symmetric(vertical: 1), child: Text("Lane $laneIndex: $bandsText ng/µL", style: const TextStyle(fontSize: 12),),);
                              },
                            ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    flex: 2,
                    child: Column(
                      children: [
                        Expanded(   // Right Top - Labeled gel image
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Container(height: 24, alignment: Alignment.centerLeft, child: const Text("Labeled gel image"),),
                              const SizedBox(height: 8),
                              Expanded(child: _annotatedBytes == null ? const Center(child: Text("NA", textAlign: TextAlign.center, ),) : Image.memory(_annotatedBytes!, fit: BoxFit.contain,),),
                            ],
                          ),
                        ),
                        const SizedBox(height: 12),
                        Expanded(   // Right Bottom - Mask image
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Container(height: 24, alignment: Alignment.centerLeft, child: const Text("Mask image (black and white)"),),
                              const SizedBox(height: 8),
                              Expanded(child: _maskBytes == null ? const Center(child: Text("NA", textAlign: TextAlign.center,),) : Image.memory(_maskBytes!, fit: BoxFit.contain,),),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          const SizedBox(height: 12),
          if (_error != null) ...[const SizedBox(height: 8), Text(_error!, style: const TextStyle(color: Colors.red, fontSize: 12),),],
        ],
      ),
    );
  }
}


//--- 4 - SETUP SAVED PAGE: RESULT (STATEFUL WIDGET, STATE)
class HistoryPage extends StatefulWidget {
  final GlobalKey<_AnalysisPageState> analysisPageKey;
  final VoidCallback onGoToResult;

  const HistoryPage({super.key, required this.analysisPageKey, required this.onGoToResult,});

  @override
  State<HistoryPage> createState() => _HistoryPageState();
}

class _HistoryPageState extends State<HistoryPage> {
  List<FileInfo> _files = [];
  bool _isLoading = true;

  @override
  void initState() {super.initState(); _loadFiles();}

// 4 - 1 - DISPLAY SAVED IMAGES FROM FILE: Saved_images
  Future<void> _loadFiles() async {
    setState(() => _isLoading = true);

    try {
      final appDir = await getApplicationDocumentsDirectory();
      final imageDir = Directory('${appDir.path}/Saved_images');
      List<FileInfo> files = [];    // creates an empty list to store FileInfo objects
      if (imageDir.existsSync()) {
        final imageFiles = imageDir.listSync();
        for (var file in imageFiles) {
          if (file is File && (file.path.endsWith('.png') || file.path.endsWith('.jpg') || file.path.endsWith('.jpeg') || file.path.endsWith('.tif'))) {files.add(FileInfo(file: file, type: FileType.image, name: file.path.split('/').last,));}
        }
      }
      files.sort((a, b) => b.file.statSync().modified.compareTo(a.file.statSync().modified));
      setState(() {_files = files; _isLoading = false;});
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Error loading files: $e")));
      setState(() => _isLoading = false);
    }
  }

// 4 - 1 - 1 ACTION - MENU DELETE FILE DISPLAYED
  Future<void> _deleteFile(FileInfo fileInfo) async {
    try {
      await fileInfo.file.delete();
      await _loadFiles();
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("File deleted")));
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Error deleting file: $e")));
    }
  }

// 4 - 1 - 2 BUILD - MENU MEASURE DNA & BUILD MEASURE DNA DIALOG
  Future<void> _showQuantifyDialog(FileInfo fileInfo) async {
    final standardController = TextEditingController();
    if (!mounted) return;
    await showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text("Measure DNA Concentration"),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text("Image: ${fileInfo.name}"),
            const SizedBox(height: 16),
            TextField(
              controller: standardController,
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
              decoration: const InputDecoration(border: OutlineInputBorder(), labelText: "Lane 1 standard DNA concentration (ng/µL)",),
            ),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text("Cancel"),),
          ElevatedButton(
            onPressed: () async {
              final standardText = standardController.text.trim();
              if (standardText.isEmpty) {
                ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Enter standard concentration")),);
                return;
              }

              final standardValue = double.tryParse(standardText);
              if (standardValue == null) {
                ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Standard concentration must be numeric")),);
                return;
              }

              Navigator.pop(context);
              if (!mounted) return;
              ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Analyzing...")),);
              await _quantifyDNA(fileInfo.file, standardValue);
            }, child: const Text("Analyze"),
          ),
        ],
      ),
    );
    standardController.dispose();
  }

  // 4 - 1 - 3 ACTION - ANALYZE IMAGE FROM SAVED FOLDER
  Future<void> _quantifyDNA(File imageFile, double standardConc) async {
    try {
      final String analyzeUrl = "http://127.0.0.1:8000/analyze";
      final String analyzeJsonUrl = "http://127.0.0.1:8000/analyze_json";

      // Extract filenames
      final imgName = imageFile.path.split(Platform.pathSeparator).last;
      final base = imgName.split('.').first;
      final ext = imgName.split('.').last;
      final timePart = base.startsWith('Gel_') ? base.substring('Gel_'.length) : base;

      print("DEBUG: ========== STARTING SAVED IMAGE ANALYSIS ==========");
      print("DEBUG: timePart = $timePart");

      // Create Reports/Data_<timePart>
      final appDir = await getApplicationDocumentsDirectory();
      final reportRoot = Directory('${appDir.path}/Reports');
      if (!reportRoot.existsSync()) reportRoot.createSync(recursive: true);
      final reportDir = Directory('${reportRoot.path}/Data_$timePart');
      if (!reportDir.existsSync()) reportDir.createSync(recursive: true);

      // Copy original gel image to Reports folder
      final gelPath = "${reportDir.path}/Gel_$timePart.$ext";
      if (!File(gelPath).existsSync()) {
        await imageFile.copy(gelPath);
      }
      print("DEBUG: Gel path: $gelPath");

      // ========== CLEAR OLD ANALYSIS PAGE DATA FIRST ==========
      final analysisState = widget.analysisPageKey.currentState;
      if (analysisState != null) {
        analysisState.setState(() {
          analysisState._laneConcs = null;
          analysisState._annotatedBytes = null;
          analysisState._maskBytes = null;
        });
        print("DEBUG: Cleared old analysis data");
      }

      // ========== CALL /analyze ENDPOINT (generates PDF + all images) ==========
      print("DEBUG: Calling /analyze endpoint...");
      final analyzeUri = Uri.parse(analyzeUrl);
      final analyzeRequest = http.MultipartRequest("POST", analyzeUri)
        ..fields["standard_conc"] = standardConc.toString()
        ..files.add(await http.MultipartFile.fromPath("file", gelPath));
      final analyzeResponse = await analyzeRequest.send();

      print("DEBUG: /analyze response status: ${analyzeResponse.statusCode}");

      if (analyzeResponse.statusCode != 200) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Analysis error: ${analyzeResponse.statusCode}")),
        );
        return;
      }

      final pdfBytes = await analyzeResponse.stream.toBytes();
      final pdfPath = "${reportDir.path}/Report_$timePart.pdf";
      await File(pdfPath).writeAsBytes(pdfBytes);
      print("DEBUG: PDF saved to $pdfPath (${pdfBytes.length} bytes)");

      // Wait a moment for backend to finish writing all files
      await Future.delayed(const Duration(milliseconds: 500));

      // ========== DOWNLOAD ANNOTATED IMAGE FROM BACKEND ==========
      print("DEBUG: Downloading annotated image...");
      Uint8List? annotatedBytes;
      try {
        final annotateUri = Uri.parse("http://127.0.0.1:8000/get_annotated/$timePart");
        print("DEBUG: Calling $annotateUri");
        final annotateResponse = await http.get(annotateUri);
        print("DEBUG: /get_annotated response status: ${annotateResponse.statusCode}");
        
        if (annotateResponse.statusCode == 200) {
          annotatedBytes = annotateResponse.bodyBytes;
          final annotatedPath = '${reportDir.path}/Annotated_$timePart.png';
          await File(annotatedPath).writeAsBytes(annotatedBytes);
          print("DEBUG: ✓ Annotated image saved to $annotatedPath (${annotatedBytes.length} bytes)");
        } else {
          print("DEBUG: ✗ Could not download annotated image: ${annotateResponse.statusCode}");
          print("DEBUG: Response body: ${annotateResponse.body}");
        }
      } catch (e) {
        print("DEBUG: ✗ Exception downloading annotated: $e");
      }

      // ========== DOWNLOAD MASK IMAGE FROM BACKEND ==========
      print("DEBUG: Downloading mask image...");
      Uint8List? maskBytes;
      try {
        final maskUri = Uri.parse("http://127.0.0.1:8000/get_mask/$timePart");
        print("DEBUG: Calling $maskUri");
        final maskResponse = await http.get(maskUri);
        print("DEBUG: /get_mask response status: ${maskResponse.statusCode}");
        
        if (maskResponse.statusCode == 200) {
          maskBytes = maskResponse.bodyBytes;
          final maskPath = '${reportDir.path}/Mask_$timePart.png';
          await File(maskPath).writeAsBytes(maskBytes);
          print("DEBUG: ✓ Mask image saved to $maskPath (${maskBytes.length} bytes)");
        } else {
          print("DEBUG: ✗ Could not download mask image: ${maskResponse.statusCode}");
          print("DEBUG: Response body: ${maskResponse.body}");
        }
      } catch (e) {
        print("DEBUG: ✗ Exception downloading mask: $e");
      }

      // ========== GET LANE CONCENTRATIONS ==========
      print("DEBUG: Getting lane concentrations...");
      List<List<double>>? lanes;
      try {
        final jsonUri = Uri.parse(analyzeJsonUrl);
        final jsonRequest = http.MultipartRequest("POST", jsonUri)
          ..fields["standard_conc"] = standardConc.toString()
          ..files.add(await http.MultipartFile.fromPath("file", gelPath));
        final jsonResponse = await http.Response.fromStream(await jsonRequest.send());

        print("DEBUG: /analyze_json response status: ${jsonResponse.statusCode}");

        if (jsonResponse.statusCode == 200) {
          final data = jsonDecode(jsonResponse.body) as Map<String, dynamic>;
          final raw = data["lane_concs"] as List;
          lanes = raw.map<List<double>>((lane) {
            if (lane is List) {
              return lane.map<double>((e) => (e as num).toDouble()).toList();
            } else if (lane is num) {
              return [lane.toDouble()];
            }
            return <double>[];
          }).toList();
          print("DEBUG: ✓ Lane concentrations: $lanes");
        }
      } catch (e) {
        print("DEBUG: ✗ Error getting lane concs: $e");
      }

      // ========== UPDATE ANALYSIS PAGE WITH NEW DATA ==========
      print("DEBUG: Updating Analysis page...");
      if (analysisState == null) {
        print("DEBUG: ✗✗✗ Analysis page state is NULL!");
      } else {
        print("DEBUG: Analysis page state exists, updating...");
        analysisState.setState(() {
          analysisState._laneConcs = lanes;
          analysisState._annotatedBytes = annotatedBytes;
          analysisState._maskBytes = maskBytes;
        });
        print("DEBUG: ✓ Analysis page updated");
        print("DEBUG:   - Lanes: ${lanes != null ? '${lanes.length} lanes' : 'NULL'}");
        print("DEBUG:   - Annotated: ${annotatedBytes != null ? '${annotatedBytes.length} bytes' : 'NULL'}");
        print("DEBUG:   - Mask: ${maskBytes != null ? '${maskBytes.length} bytes' : 'NULL'}");
      }

      print("DEBUG: ========== SAVED IMAGE ANALYSIS COMPLETE ==========");

      // ========== SUCCESS ==========
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Analysis complete: $pdfPath")),
      );
      await OpenFilex.open(pdfPath);
      await _loadFiles();
      widget.onGoToResult();

    } catch (e, stackTrace) {
      print("DEBUG: ✗✗✗ FATAL ERROR in _quantifyDNA: $e");
      print("DEBUG: Stack trace: $stackTrace");
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error: $e")),
      );
    }
  }

  
// 4 - 1 - 4 - BUILD LAYOUT: SAVED TAB & DROP-DOWN MENU
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              ElevatedButton.icon(onPressed: () async{
                final appDir = await getApplicationDocumentsDirectory();
                final reportDir = Directory('${appDir.path}/Reports');
                if (!reportDir.existsSync()){reportDir.createSync(recursive: true);}
                await OpenFilex.open(reportDir.path);
              },
              icon: const Icon(Icons.folder_open), label: const Text("Open Reports"),
              ),
              ElevatedButton.icon(onPressed: _loadFiles, icon: const Icon(Icons.refresh), label: const Text("Refresh"),),
            ],
          ),
          const SizedBox(height: 16),
          Expanded(
            child: _isLoading
                ? const Center(child: CircularProgressIndicator())
                : _files.isEmpty
                    ? const Center(child: Text("No files yet"))
                    : ListView.builder(
                        itemCount: _files.length,
                        itemBuilder: (context, index) {
                          final fileInfo = _files[index];
                          return Card(
                            margin: const EdgeInsets.symmetric(vertical: 8),
                            child: ListTile(
                              leading: Icon(fileInfo.type == FileType.image ? Icons.image : Icons.picture_as_pdf, color: fileInfo.type == FileType.image ? Colors.blue : Colors.red,),
                              title: Text(fileInfo.name, overflow: TextOverflow.ellipsis),
                              subtitle: Text("Modified: ${fileInfo.file.statSync().modified.toString().split('.').first}", style: const TextStyle(fontSize: 12),),
                              trailing: PopupMenuButton(
                                itemBuilder: (context) => [
                                  PopupMenuItem(child: const Text("Open"), onTap: () => OpenFilex.open(fileInfo.file.path),),
                                  PopupMenuItem(child: const Text("Measure DNA"), onTap: () => _showQuantifyDialog(fileInfo)),
                                  PopupMenuItem(child: const Text("Delete"), onTap: () => _deleteFile(fileInfo),),
                                ],
                              ),
                              onTap: () => OpenFilex.open(fileInfo.file.path),
                            ),
                          );
                        },
                      ),
          ),
        ],
      ),
    );
  }
}

enum FileType { image, pdf }

class FileInfo {
  final File file;
  final FileType type;
  final String name;
  FileInfo({required this.file, required this.type, required this.name,});
}


