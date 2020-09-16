/*
	This program is created by referring to dnn_face_recognition_ex.cpp.
	Most of the main mechanism is the same as dnn_face_recognition_ex.cpp.

	It uses the pre-trained dlib_face_recognition_resnet_model_v1 model.

	Quoting the comments from the original dnn_face_recognition_ex.cpp, 
	the accuracy of the standard LFW surface for this model is 99.38%.
	A recognition benchmark comparable to other cutting-edge methods for the face
	Certified as of February 2017.
*/
#include "dnn_face_recognition_ex.hpp"

int main(int argc, char** argv) try
{
	if (argc < 2)
	{
		printf("%s args\n", argv[0]);
		printf("args:\n");
		printf("--cap [username]\n");
		printf("       create face image -> capture\n");
		printf("--m\n");
		printf("       imagelist.txt ->(output) shapelist.txt\n");
		printf("--recog\n");
		printf("       real time camera image -> face recognition\n");
		printf("--image imagefile[.png|.jpg]\n");
		printf("       imagefile -> face recognition\n");
		printf("\n");
		printf("imagefile[.png|.jpg] ->(output) user_shape/imagefile.txt\n");

		return -1;
	}

	int camID = 0;
	for ( int i = 2; i < argc; i++)
	{
		if (std::string(argv[i]) == "--camID")
		{
			camID = atoi(argv[i+1]);
			i++;
		}
		if (std::string(argv[i]) == "--face_chk")
		{
			dnn_face_recognition_::face_chk = atoi(argv[i+1]);
			i++;
		}
	}
	printf("camID= %d\n", camID);

	if (std::string(argv[1]) == "--m")
	{
		printf("%d\n", make_shape());
		exit(0);
	}
	if (std::string(argv[1]) == "--cap")
	{
		char* user_name = "";
		if (argc >= 3) user_name = argv[2];
		if (cam2face_shape(user_name, camID) != 0)
		{
			printf("I couldn't capture the front facing face\n");
		}
		exit(0);
	}

	if (std::string(argv[1]) == "--recog")
	{
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor sp;
		deserialize("db/shape_predictor_5_face_landmarks.dat") >> sp;
		anet_type net;
		deserialize("db/dlib_face_recognition_resnet_model_v1.dat") >> net;

		shape_predictor sp68;
		deserialize("db/shape_predictor_68_face_landmarks.dat") >> sp68;

		std::vector<std::string> shapelist;
		std::vector<std::vector<float>> shapevalue_list;
		if (shapelist.size() == 0)
		{
			if (load_shapelist(shapelist, shapevalue_list) != 0)
			{
				return -1;
			}
		}

		do {
			
			std::string& user_name = webcam_face_recognition(detector, sp, sp68, net, shapelist, shapevalue_list, camID);
			printf("user %s\n", user_name.c_str());

			if (user_name != "" && user_name != "unknown")
			{
				std::string pathname;
				std::string extname;
				std::string& filename = getFilename(user_name, pathname, extname);

				std::string img = "images/" + filename + ".png";
				try
				{
					auto user = cv::imread(img);
					if (user.empty())
					{
						img = "images/" + filename + ".jpg";
						user = cv::imread(img);
					}
					if (!user.empty())
					{
						sc::myCV::putText_Jpn(user, (filename).c_str(), cv::Point(50, 90), std::string("ÇlÇr ÉSÉVÉbÉN").c_str(), 0.5, cv::Scalar(200, 0, 0), 3);

						cv::imshow(filename, user);
						cv::waitKey(10);
					}
				}
				catch (...)
				{
				}
			}
			cout << "press any key to continue.." << endl;
			cin.get();
		} while (1);
		exit(0);
	}

	if (std::string(argv[1]) == "--image")
	{
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor sp;
		deserialize("db/shape_predictor_5_face_landmarks.dat") >> sp;
		anet_type net;
		deserialize("db/dlib_face_recognition_resnet_model_v1.dat") >> net;

		shape_predictor sp68;
		deserialize("db/shape_predictor_68_face_landmarks.dat") >> sp68;

		cv::Mat face_image = cv::imread(argv[2]);
		std::vector<std::string> shapelist;
		std::vector<std::vector<float>> shapevalue_list;

		if (!face_dir_check(face_image, detector, sp68))
		{
			printf("You are not facing the front or you can see multiple people.\n");
			return 1;
		}
		float dist;
		float cos_dist;
		std::string user_name = face_recognition(face_image, detector, sp, net, shapelist, shapevalue_list, dist, cos_dist);
		printf("user %s\n", user_name.c_str());

		if (user_name != "" && user_name != "unknown")
		{
			std::string pathname;
			std::string extname;
			std::string& filename = getFilename(user_name, pathname, extname);

			std::string img = "images/" + filename + ".png";
			try
			{
				auto user = cv::imread(img);
				if (user.empty())
				{
					img = "images/" + filename + ".jpg";
					user = cv::imread(img);
				}
				if (!user.empty())
				{
					sc::myCV::putText_Jpn(user, (filename).c_str(), cv::Point(50, 90), std::string("ÇlÇr ÉSÉVÉbÉN").c_str(), 0.5, cv::Scalar(200, 0, 0), 3);

					cv::imshow(filename, user);
					cv::waitKey(10);
				}
			}
			catch (...)
			{
			}
		}
		exit(0);
	}

	if (std::string(argv[1]) == "--test")
	{
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor sp;
		deserialize("db/shape_predictor_5_face_landmarks.dat") >> sp;
		anet_type net;
		deserialize("db/dlib_face_recognition_resnet_model_v1.dat") >> net;

		shape_predictor sp68;
		deserialize("db/shape_predictor_68_face_landmarks.dat") >> sp68;

		std::vector<std::string> shapelist;
		std::vector<std::vector<float>> shapevalue_list;

		if (shapelist.size() == 0)
		{
			if (load_shapelist(shapelist, shapevalue_list) != 0)
			{
				return -1;
			}
		}

		FILE* fp = fopen("test.csv", "w");
		fprintf(fp, "face,predict,dist,cos_dist\n");

		int count = 0;
		int ok = 0;
		printf("shapelist=%d\n", shapelist.size());
		for (int i = 0; i < shapelist.size(); i++)
		{
			std::string pathname;
			std::string extname;
			std::string& filename = getFilename(shapelist[i], pathname, extname);

			cv::Mat face_image;
			std::string img = "images/" + filename + ".png";
			try
			{
				face_image = cv::imread(img);
				if (face_image.empty())
				{
					img = "images/" + filename + ".jpg";
					face_image = cv::imread(img);
				}
				if (face_image.empty())
				{
					continue;
				}
				if (!face_dir_check(face_image, detector, sp68))
				{
					printf("You are not facing the front or you can see multiple people.\n");
					cv::imwrite("tmp/error_" + std::to_string(i) + " .png", face_image);
					continue;
				}
			}
			catch (...)
			{
				continue;
			}

			count++;
			printf("user %s\n", shapelist[i].c_str()); fflush(stdout);
			//fprintf(fp, "%s", shapelist[i].c_str());
			fprintf(fp, "%s", getUserName(shapelist[i].c_str()).c_str());

			std::vector<float> org(shapevalue_list[i].size());
			std::vector<float> del(shapevalue_list[i].size());
			org = shapevalue_list[i];

			if (i > 0)
			{
				std::string a = getUserName(shapelist[i - 1].c_str());
				std::string b = getUserName(shapelist[i].c_str());
				if (a == b)
				{
					shapevalue_list[i] = del;
				}
				if (i < shapelist.size() - 1)
				{
					std::string c = getUserName(shapelist[i + 1].c_str());
					if (b == c)
					{
						shapevalue_list[i] = del;
					}
				}
			}

			float dist;
			float cos_dist;
			std::string user_name = face_recognition(face_image, detector, sp, net, shapelist, shapevalue_list, dist, cos_dist);
			shapevalue_list[i] = org;
			printf("user %s\n", user_name.c_str());
			fprintf(fp, ",%s,%f,%f,%d\n", getUserName(user_name.c_str()).c_str(), dist, cos_dist, getUserName(shapelist[i].c_str()) == getUserName(user_name.c_str())?1:0);

			if (user_name != "" && user_name != "unknown")
			{
				std::string pathname;
				std::string extname;
				std::string& filename = getFilename(user_name, pathname, extname);

				std::string img = "images/" + filename + ".png";
				auto user = cv::imread(img);
				if (user.empty())
				{
					img = "images/" + filename + ".jpg";
					user = cv::imread(img);
				}
				if (!user.empty())
				{
					sc::myCV::putText_Jpn(user, (filename).c_str(), cv::Point(10, 90), std::string("ÇlÇr ÉSÉVÉbÉN").c_str(), 0.5, cv::Scalar(150, 255, 255), 3);
				}

				if (getUserName(shapelist[i].c_str()) == getUserName(user_name.c_str()))
				{
					ok++;
				}
				else
				{
					std::string& filename = getFilename(shapelist[i], pathname, extname);
					std::string img = "images/" + filename + ".png";
					auto tmp = cv::imread(img);
					if (tmp.empty())
					{
						img = "images/" + filename + ".jpg";
						tmp = cv::imread(img);
					}
					if (!user.empty())
					{
						sc::myCV::putText_Jpn(tmp, (filename).c_str(), cv::Point(10, 90), std::string("ÇlÇr ÉSÉVÉbÉN").c_str(), 0.5, cv::Scalar(150, 255, 255), 3);
					}
					cv::Mat cat = hconcat_ex(tmp, user);
					cv::imwrite("tmp/error_"+ std::to_string(i) +" .png", cat);

					cv::imshow("-", cat);
					cv::waitKey(10);
					//cv::imwrite("tmp/error" + std::to_string(2*i) + " .png", tmp);
					//cv::imwrite("tmp/error" + std::to_string(2*i+1) + " .png", user);
				}
			}
			//cin.get();
			printf("%d/%d %d (%.3f)\n", ok, count, shapelist.size(), 100.0*(float)ok / (float)count);
		}
		fprintf(fp, "%d,%d,%f\n", count, ok, 0);
		fclose(fp);
		exit(0);
	}


    // The first thing we are going to do is load all our models.  First, since we need to
    // find faces in the image we will need a face detector:
    frontal_face_detector detector = get_frontal_face_detector();
    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
    shape_predictor sp;
    deserialize("db/shape_predictor_5_face_landmarks.dat") >> sp;
    // And finally we load the DNN responsible for face recognition.
    anet_type net;
    deserialize("db/dlib_face_recognition_resnet_model_v1.dat") >> net;

    matrix<rgb_pixel> img;
	try
	{
		load_image(img, argv[1]);
	}
	catch (...)
	{
		cout << "load_image() Error." << endl;
		return -2;
	}

    // Display the raw image on the screen
    //image_window win(img); 

    // Run the face detector on the image of our action heroes, and for each face extract a
    // copy that has been normalized to 150x150 pixels in size and appropriately rotated
    // and centered.
    std::vector<matrix<rgb_pixel>> faces;
    for (auto face : detector(img))
    {
        auto shape = sp(img, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
        faces.push_back(move(face_chip));
	}

    if (faces.size() == 0)
    {
        cout << "No faces found in image!" << endl;
        return 1;
    }
	if (faces.size() > 1)
	{
		cout << "faces > 1 in image!" << endl;
		return 2;
	}

    // This call asks the DNN to convert each face image in faces into a 128D vector.
    // In this 128D vector space, images from the same person will be close to each other
    // but vectors from different people will be far apart.  So we can use these vectors to
    // identify if a pair of images are from the same person or from different people.  
    std::vector<matrix<float,0,1>> face_descriptors = net(faces);


    // In particular, one simple thing we can do is face clustering.  This next bit of code
    // creates a graph of connected faces and then uses the Chinese whispers graph clustering
    // algorithm to identify how many people there are and which faces belong to whom.
    std::vector<sample_pair> edges;
    for (size_t i = 0; i < face_descriptors.size(); ++i)
    {
        for (size_t j = i; j < face_descriptors.size(); ++j)
        {
            // Faces are connected in the graph if they are close enough.  Here we check if
            // the distance between two face descriptors is less than 0.6, which is the
            // decision threshold the network was trained to use.  Although you can
            // certainly use any other threshold you find useful.
            if (length(face_descriptors[i]-face_descriptors[j]) < 0.6)
                edges.push_back(sample_pair(i,j));
        }
    }
    std::vector<unsigned long> labels;
    const auto num_clusters = chinese_whispers(edges, labels);
    // This will correctly indicate that there are 4 people in the image.
    cout << "number of people found in the image: "<< num_clusters << endl;


	std::string fullpath = std::string(argv[1]);
	int path_i = 0;
	if ( strchr(fullpath.c_str(), '\\')) path_i = fullpath.find_last_of("\\") + 1;//7
	else path_i = fullpath.find_last_of("/") + 1;//7

	int ext_i = fullpath.find_last_of(".");//10
	std::string pathname = fullpath.substr(0, path_i + 1);//0ï∂éöñ⁄Ç©ÇÁÇVï∂éöêÿÇËèoÇ∑ "C:\\aaa\\"
	std::string extname = fullpath.substr(ext_i, fullpath.size() - ext_i); // 10ï∂éöñ⁄Ç©ÇÁÇSï∂éöêÿÇËèoÇ∑ ".txt"
	std::string filename = fullpath.substr(path_i, ext_i - path_i);// ÇVï∂éöñ⁄Ç©ÇÁÇRï∂éöêÿÇËèoÇ∑Å@"bbb"

    // Now let's display the face clustering results on the screen.  You will see that it
    // correctly grouped all the faces. 
    std::vector<image_window> win_clusters(num_clusters);
    for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
    {
        std::vector<matrix<rgb_pixel>> temp;
        for (size_t j = 0; j < labels.size(); ++j)
        {
            if (cluster_id == labels[j])
                temp.push_back(faces[j]);
        }
        win_clusters[cluster_id].set_title("face cluster " + cast_to_string(cluster_id));
        win_clusters[cluster_id].set_image(tile_images(temp));

		char clipped[256];
		sprintf(clipped, "user_images/%s.png", filename.c_str());
		cv::Mat x = dlib::toMat(tile_images(temp)).clone();
		cv::cvtColor(x, x, CV_RGB2BGR);
		cv::imwrite(clipped, x);
		break;
	}


    // Finally, let's print one of the face descriptors to the screen.  
    cout << "face descriptor for one face: " << trans(face_descriptors[0]) << endl;

    // It should also be noted that face recognition accuracy can be improved if jittering
    // is used when creating face descriptors.  In particular, to get 99.38% on the LFW
    // benchmark you need to use the jitter_image() routine to compute the descriptors,
    // like so:
    matrix<float,0,1> face_descriptor = mean(mat(net(jitter_image(faces[0]))));
    cout << "jittered face descriptor for one face: " << trans(face_descriptor) << endl;
    // If you use the model without jittering, as we did when clustering the bald guys, it
    // gets an accuracy of 99.13% on the LFW benchmark.  So jittering makes the whole
    // procedure a little more accurate but makes face descriptor calculation slower.

	char user_shape[256];
	sprintf(user_shape, "user_shape/%s.txt", filename.c_str());
	printf("%s\n", user_shape);

	FILE* fp = fopen(user_shape, "w");
	if (fp)
	{
		auto& t = face_descriptor;
		for (int i = 0; i < t.size(); i++)
		{
			fprintf(fp, "%f\n", t(i));
		}
		//for (int i = 0; i < t.nc(); i++)
		//{
		//	for (int j = 0; j < t.nr(); j++)
		//	{
		//		fprintf(fp, "%f\n", t(i, j));
		//	}
		//}
		fclose(fp);
	}
    //cout << "hit enter to terminate" << endl;
    //cin.get();
}
catch (std::exception& e)
{
    cout << e.what() << endl;
}

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
)
{
    // All this function does is make 100 copies of img, all slightly jittered by being
    // zoomed, rotated, and translated a little bit differently. They are also randomly
    // mirrored left to right.
    thread_local dlib::rand rnd;

    std::vector<matrix<rgb_pixel>> crops; 
    for (int i = 0; i < 100; ++i)
        crops.push_back(jitter_image(img,rnd));

    return crops;
}

// ----------------------------------------------------------------------------------------

