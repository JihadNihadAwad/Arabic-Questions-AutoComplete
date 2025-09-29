package ai.project.gui;

import java.io.InputStream;

import javafx.fxml.FXMLLoader;
import javafx.scene.image.Image;
import javafx.scene.layout.Background;
import javafx.scene.layout.BackgroundImage;
import javafx.scene.layout.BackgroundPosition;
import javafx.scene.layout.BackgroundRepeat;
import javafx.scene.layout.BackgroundSize;

/**
 * A utility class for loading and accessing resources such as FXML files, CSS
 * stylesheets, and images.
 */
public class ResourceManager {
	/**
	 * Private constructor to prevent instantiation.
	 */
	private ResourceManager() {
	}

	/**
	 * Returns a FXMLLoader instance for the specified FXML file.
	 * 
	 * @param name the name of the FXML file, without the .fxml extension
	 * @return a FXMLLoader instance
	 */
	public static FXMLLoader getFXMLLoader(String name) {
		String url = "/fxml/" + name;
		if (!url.contains(".fxml")) {
			url = url + ".fxml";
		}
		return new FXMLLoader(ResourceManager.class.getResource(url));
	}

	/**
	 * Returns the URL of the specified CSS stylesheet as a string.
	 * 
	 * @param name the name of the CSS file, without the .css extension
	 * @return the URL of the CSS file
	 */
	public static String getStylesheetURL(String name) {
		String url = "/css/" + name;
		if (!url.contains(".css")) {
			url = url + ".css";
		}
		return ResourceManager.class.getResource(url).toExternalForm();
	}

	/**
	 * Returns an Image instance for the specified image file.
	 * 
	 * @param name the name of the image file, without the file extension
	 * @return an Image instance, or null if the image file is not found
	 */
	public static Image getImage(String name) {
		String url = "/images/" + name;
		if (!url.contains(".")) {
			url = url + ".png";
		}
		InputStream is = ResourceManager.class.getResourceAsStream(url);
		if (is == null)
			return null;
		return new Image(is);
	}
	
	/**	
	 * Returns a background instance for the specified image file.
	 * @param name the name of the image file
	 * @return a background instance
	 */
	public static Background getBackground(String name) {
		Image image = ResourceManager.getImage(name);
		BackgroundImage backgroundImage = new BackgroundImage(image, BackgroundRepeat.NO_REPEAT,
				BackgroundRepeat.NO_REPEAT, BackgroundPosition.CENTER,
				new BackgroundSize(1.0, 1.0, true, true, true, true));

		return new Background(backgroundImage);
	}
}
